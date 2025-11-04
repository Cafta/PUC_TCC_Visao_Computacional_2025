import cv2
import mediapipe as mp
import numpy as np
from enum import Enum
import time
from collections import deque
import libs.home_assistant_lib as ha_libs

# ==================== CONFIGURAÇÕES ====================
class Config:
    # Câmera - RESOLUÇÃO REDUZIDA para melhor FPS
    CAMERA_INDEX = 0
    FRAME_WIDTH = 1280  # Reduzido de 1280
    FRAME_HEIGHT = 720  # Reduzido de 720
    
    # Performance - SKIP FRAMES para economizar processamento
    PROCESS_EVERY_N_FRAMES = 4  # Processa 1 a cada 2 frames (dobra FPS)
    
    # GPU Settings
    USE_GPU = True  # Tenta usar GPU se disponível
    CUDA_DEVICE_ID = 0  # ID da GPU (0 para primeira)
    
    # Detecção de movimento - OTIMIZADO
    MOG2_HISTORY = 300  # Reduzido de 500
    MOG2_VAR_THRESHOLD = 20  # Aumentado para menos sensibilidade
    MIN_CONTOUR_AREA = 3000  # Reduzido para detectar melhor
    USE_CUDA_MOG2 = False  # MOG2 em GPU (requer OpenCV+CUDA)
    
    # MediaPipe settings - MODO LITE para máxima velocidade
    POSE_MODEL_COMPLEXITY = 0  # 0=lite (RÁPIDO), 1=full, 2=heavy
    HAND_MODEL_COMPLEXITY = 0  # 0=lite (RÁPIDO), 1=full
    POSE_CONFIDENCE = 0.5
    HAND_CONFIDENCE = 0.5
    
    # Pose detection
    ARM_RAISED_THRESHOLD = -0.1  # Pulso acima do ombro (y invertido)
    
    # Hand detection
    ROI_MARGIN = 1.5  # Margem ao redor da mão
    
    # Gesture recognition - AJUSTADO pelo usuário
    OPEN_HAND_THRESHOLD = 0.70  # Mão aberta
    CLOSED_HAND_THRESHOLD = 0.30  # Mão fechada
    GESTURE_HOLD_TIME = 1.0  # Segundos para confirmar gesto (ajustado)
    QUICK_TRANSITION_TIME = 0.3  # Tempo máximo para transição rápida (ajustado)
    STABILITY_THRESHOLD = 0.08  # Variação máxima para considerar estável
    COOLDOWN_TIME = 1.0  # Tempo após comando (ajustado)
    IDLE_RESET_TIME = 0.8  # Tempo mantendo estado antes de poder fazer novo gesto
    
    # Calibration mode
    CALIBRATION_MODE = False  # Ativa modo de calibração (pressione 'c')
    
    # Visual feedback
    SHOW_DEBUG = True
    FPS_AVERAGE_FRAMES = 30


# ==================== DETECÇÃO DE GPU ====================
class GPUInfo:
    """Classe para detectar e gerenciar uso de GPU"""
    
    @staticmethod
    def check_cuda_available():
        """Verifica se CUDA está disponível"""
        try:
            # Verifica OpenCV CUDA
            opencv_cuda = cv2.cuda.getCudaEnabledDeviceCount() > 0
        except:
            opencv_cuda = False
        
        return opencv_cuda
    
    @staticmethod
    def check_mediapipe_gpu():
        """Verifica se MediaPipe pode usar GPU"""
        try:
            # MediaPipe usa GPU automaticamente se CUDA estiver disponível
            # Não há API direta para verificar, mas podemos inferir
            import subprocess
            result = subprocess.run(['nvidia-smi'], 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=2)
            return result.returncode == 0
        except:
            return False
    
    @staticmethod
    def print_gpu_info():
        """Imprime informações sobre GPU disponível"""
        print("\n" + "="*60)
        print("🔍 INFORMAÇÕES DE GPU")
        print("="*60)
        
        # OpenCV CUDA
        opencv_cuda = GPUInfo.check_cuda_available()
        print(f"OpenCV com CUDA: {'✅ Disponível' if opencv_cuda else '❌ Não disponível'}")
        
        if opencv_cuda:
            count = cv2.cuda.getCudaEnabledDeviceCount()
            print(f"Dispositivos CUDA detectados: {count}")
            for i in range(count):
                print(f"  GPU {i}: {cv2.cuda.getDevice()}")
        
        # MediaPipe GPU
        mp_gpu = GPUInfo.check_mediapipe_gpu()
        print(f"MediaPipe GPU: {'✅ Detectada' if mp_gpu else '❌ Não disponível'}")
        
        # Recomendações
        print("\n💡 IMPORTANTE:")
        print("  ⚠️  MediaPipe Python usa GPU apenas para renderização OpenGL")
        print("     A inferência (detecção) roda em CPU com TensorFlow Lite")
        print("     Isso é uma limitação do MediaPipe Python API")
        
        if not opencv_cuda and Config.USE_GPU:
            print("\n  ⚠️  OpenCV não foi compilado com CUDA")
            print("     Para melhor performance, recompile OpenCV com CUDA")
        
        print("\n📊 PERFORMANCE ESPERADA:")
        print("  • Resolução 640x480 + Model Lite: 18-25 FPS")
        print("  • Resolução 1280x720 + Model Full: 8-12 FPS")
        print("  • Para >30 FPS: Use resolução menor ou skip frames")
        
        print("="*60 + "\n")


# ==================== ESTADOS DO SISTEMA ====================
class SystemState(Enum):
    IDLE = 1              # Aguardando movimento
    MONITORING = 2        # Pessoa detectada, monitorando pose
    ARM_RAISED = 3        # Braço levantado, tracking mão
    GESTURE_DETECTION = 4 # Analisando gesto abrir/fechar
    COOLDOWN = 5          # Aguardando antes de aceitar novo gesto


# ==================== DETECTOR DE MOVIMENTO ====================
class MotionDetector:
    def __init__(self, use_cuda=False):
        self.use_cuda = use_cuda and GPUInfo.check_cuda_available()
        
        if self.use_cuda:
            print("🚀 MOG2: Usando GPU (CUDA)")
            # OpenCV CUDA MOG2
            self.bg_subtractor = cv2.cuda.createBackgroundSubtractorMOG2(
                history=Config.MOG2_HISTORY,
                varThreshold=Config.MOG2_VAR_THRESHOLD,
                detectShadows=True
            )
            self.gpu_frame = cv2.cuda_GpuMat()
            self.gpu_mask = cv2.cuda_GpuMat()
        else:
            print("💻 MOG2: Usando CPU")
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=Config.MOG2_HISTORY,
                varThreshold=Config.MOG2_VAR_THRESHOLD,
                detectShadows=True
            )
        
        self.motion_detected = False
        
    def detect(self, frame):
        """Detecta movimento no frame"""
        if self.use_cuda:
            # Upload para GPU
            self.gpu_frame.upload(frame)
            
            # Aplica background subtraction na GPU
            self.bg_subtractor.apply(self.gpu_frame, self.gpu_mask, -1)
            
            # Download de volta para CPU
            fg_mask = self.gpu_mask.download()
        else:
            fg_mask = self.bg_subtractor.apply(frame)
        
        # Remove sombras
        fg_mask[fg_mask == 127] = 0
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Encontra contornos
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        # Verifica se há movimento significativo
        self.motion_detected = any(
            cv2.contourArea(c) > Config.MIN_CONTOUR_AREA for c in contours
        )
        
        return self.motion_detected, fg_mask


# ==================== DETECTOR DE POSE ====================
class PoseDetector:
    def __init__(self, use_gpu=True):
        self.mp_pose = mp.solutions.pose
        
        # MediaPipe automaticamente usa GPU se disponível
        # model_complexity: 0 (lite/rápido), 1 (full), 2 (heavy/preciso)
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=Config.POSE_MODEL_COMPLEXITY,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=Config.POSE_CONFIDENCE,
            min_tracking_confidence=Config.POSE_CONFIDENCE
        )
        
        print(f"🚀 MediaPipe Pose: Complexity={Config.POSE_MODEL_COMPLEXITY} "
              f"({'GPU se disponível' if use_gpu else 'CPU'})")
        
        self.person_detected = False
        self.arm_raised = False
        self.raised_hand_position = None
        
    def detect(self, frame):
        """Detecta pessoa e verifica se braço está levantado"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        self.person_detected = results.pose_landmarks is not None
        self.arm_raised = False
        self.raised_hand_position = None
        arm_length = None  # Distância punho-cotovelo para normalização
        
        if self.person_detected:
            landmarks = results.pose_landmarks.landmark
            
            # Verifica braço direito levantado
            right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]
            right_elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
            
            # Verifica braço esquerdo levantado
            left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
            left_elbow = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW]
            
            # Braço direito levantado?
            if (right_wrist.y < right_shoulder.y + Config.ARM_RAISED_THRESHOLD and
                right_wrist.visibility > 0.5):
                self.arm_raised = True
                self.raised_hand_position = (right_wrist.x, right_wrist.y)
                # Calcula distância punho-cotovelo (escala de referência)
                arm_length = np.sqrt(
                    (right_wrist.x - right_elbow.x)**2 + 
                    (right_wrist.y - right_elbow.y)**2
                )
                
            # Braço esquerdo levantado?
            elif (left_wrist.y < left_shoulder.y + Config.ARM_RAISED_THRESHOLD and
                  left_wrist.visibility > 0.5):
                self.arm_raised = True
                self.raised_hand_position = (left_wrist.x, left_wrist.y)
                # Calcula distância punho-cotovelo (escala de referência)
                arm_length = np.sqrt(
                    (left_wrist.x - left_elbow.x)**2 + 
                    (left_wrist.y - left_elbow.y)**2
                )
        
        return self.person_detected, self.arm_raised, self.raised_hand_position, arm_length, results


# ==================== DETECTOR DE MÃOS ====================
class HandDetector:
    def __init__(self, use_gpu=True):
        self.mp_hands = mp.solutions.hands
        
        # MediaPipe Hands com configuração otimizada
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=Config.HAND_MODEL_COMPLEXITY,
            min_detection_confidence=Config.HAND_CONFIDENCE,
            min_tracking_confidence=Config.HAND_CONFIDENCE
        )
        
        print(f"🚀 MediaPipe Hands: Complexity={Config.HAND_MODEL_COMPLEXITY} "
              f"({'GPU se disponível' if use_gpu else 'CPU'})")
        
        self.hand_detected = False
        self.hand_openness = 0.0
        self.hand_center = None
        
    def get_roi(self, frame, hand_position, arm_length=None):
        """
        Calcula ROI ao redor da posição da mão.
        Ajusta o tamanho baseado no comprimento do braço (escala).
        """
        h, w = frame.shape[:2]
        x, y = int(hand_position[0] * w), int(hand_position[1] * h)
        
        # Se temos arm_length, usa ele para ajustar o tamanho do ROI
        # Caso contrário, usa tamanho padrão
        if arm_length and arm_length > 0:
            # ROI proporcional ao tamanho do braço
            # arm_length está normalizado (0-1), então multiplicamos por dimensão da imagem
            roi_size = int(arm_length * max(w, h) * 1)  # 2.5x o comprimento do antebraço
            roi_size = max(100, min(roi_size, min(w, h)))  # Limita entre 100px e tamanho da imagem
        else:
            # Tamanho padrão fixo
            roi_size = int(min(w, h) * 0.25)
        
        margin = int(roi_size * Config.ROI_MARGIN)
        
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(w, x + margin)
        y2 = min(h, y + margin)
        
        return (x1, y1, x2, y2)
    
    def calculate_openness(self, landmarks, arm_length=None):
        """
        Calcula o quão aberta está a mão (0=fechada, 1=aberta)
        Usa arm_length (distância punho-cotovelo) para normalizar e ser independente da distância
        CALIBRADO com valores reais observados
        """
        wrist = landmarks[0]
        
        # Se não temos arm_length, tenta estimar pela própria mão
        if arm_length is None or arm_length <= 0.001:
            # Usa distância do pulso até a base do dedo médio como referência
            middle_base = landmarks[9]
            arm_length = np.sqrt(
                (middle_base.x - wrist.x)**2 + 
                (middle_base.y - wrist.y)**2
            )
            # Proteção contra divisão por zero
            if arm_length <= 0.001:
                arm_length = 0.1
        
        # Método 1: Distância normalizada das pontas dos dedos ao pulso
        finger_tips = [4, 8, 12, 16, 20]  # Polegar, indicador, médio, anelar, mínimo
        tip_distances = []
        
        for tip_idx in finger_tips:
            tip = landmarks[tip_idx]
            distance = np.sqrt(
                (tip.x - wrist.x)**2 + 
                (tip.y - wrist.y)**2
            )
            # NORMALIZA pela distância punho-cotovelo!
            normalized_distance = distance / arm_length
            tip_distances.append(normalized_distance)
        
        avg_tip_distance = np.mean(tip_distances)
        
        # Método 2: Spread dos dedos normalizado
        index_tip = landmarks[8]
        pinky_tip = landmarks[20]
        finger_spread = np.sqrt(
            (index_tip.x - pinky_tip.x)**2 + 
            (index_tip.y - pinky_tip.y)**2
        )
        normalized_spread = finger_spread / arm_length
        
        # Método 3: Extensão dos dedos (razão, já é independente de escala)
        finger_extensions = []
        finger_pairs = [
            (8, 6),   # Indicador: ponta vs junta média
            (12, 10), # Médio: ponta vs junta média
            (16, 14), # Anelar: ponta vs junta média
            (20, 18), # Mínimo: ponta vs junta média
        ]
        
        for tip_idx, middle_idx in finger_pairs:
            tip = landmarks[tip_idx]
            middle = landmarks[middle_idx]
            
            tip_to_wrist = np.sqrt((tip.x - wrist.x)**2 + (tip.y - wrist.y)**2)
            middle_to_wrist = np.sqrt((middle.x - wrist.x)**2 + (middle.y - wrist.y)**2)
            
            if middle_to_wrist > 0.01:
                extension = tip_to_wrist / middle_to_wrist
                finger_extensions.append(extension)
        
        avg_extension = np.mean(finger_extensions) if finger_extensions else 1.0
        
        # DEBUG: Imprime valores brutos se em modo calibração
        if Config.CALIBRATION_MODE:
            print(f"[CALIB] tip_dist={avg_tip_distance:.3f}, spread={normalized_spread:.3f}, ext={avg_extension:.3f}, arm_len={arm_length:.3f}")
        
        # ===== CALIBRAÇÃO BASEADA EM VALORES REAIS OBSERVADOS =====
        # Dados da calibração do usuário:
        CLOSED_TIP_DIST = 0.75  # Era 0.30, mas sua mão fechada é 0.35
        OPEN_TIP_DIST = 1.4    # O valor que faz a mão ser UM (0.62-0.69)

        CLOSED_SPREAD = 0.45    # Era 0.18, mas sua mão fechada é 0.23
        OPEN_SPREAD = 0.8      # O valor que faz a mão ser UM (0.38-0.41)

        CLOSED_EXTENSION = 0.65 # Mantido, pois é o mais confiável
        OPEN_EXTENSION = 1.15   # Era 1.35

        # Subtrai o valor "fechado" e divide pela amplitude (aberto - fechado)
        tip_score = np.clip((avg_tip_distance - CLOSED_TIP_DIST) / (OPEN_TIP_DIST - CLOSED_TIP_DIST), 0, 1)
        spread_score = np.clip((normalized_spread - CLOSED_SPREAD) / (OPEN_SPREAD - CLOSED_SPREAD), 0, 1)
        extension_score = np.clip((avg_extension - CLOSED_EXTENSION) / (OPEN_EXTENSION - CLOSED_EXTENSION), 0, 1)


        # # Fechada: tip_dist=0.32-0.35, spread=0.19-0.23, ext=0.69-0.78
        # # Aberta:  tip_dist=0.62-0.69, spread=0.38-0.41, ext=1.30-1.34
        
        # # Normaliza cada métrica para 0-1 usando valores observados
        # tip_score = np.clip((avg_tip_distance - 0.30) / 0.38, 0, 1)
        # spread_score = np.clip((normalized_spread - 0.18) / 0.23, 0, 1)
        # extension_score = np.clip((avg_extension - 0.68) / 0.67, 0, 1)
        
        # Média ponderada (extensão mais confiável, depois tip, depois spread)
        openness = (extension_score * 0.5 + tip_score * 0.35 + spread_score * 0.15)
        
        return float(openness)
    
    def detect(self, frame, roi=None, arm_length=None):
        """
        Detecta mão no frame ou ROI. Inclui ZOÓM no ROI para melhor escala.
        """
        if roi:
            x1, y1, x2, y2 = roi
            roi_frame = frame[y1:y2, x1:x2].copy()
            if roi_frame.size == 0:
                return False, 0.0, None
        else:
            roi_frame = frame.copy()
            x1, y1 = 0, 0
        
        # 💡 PASSO CRÍTICO: Redimensiona o ROI (Zoom)
        # Força o ROI a ter um tamanho fixo e grande para 'ampliar' a mão,
        # garantindo que o objeto de interesse tenha muitos pixels.
        TARGET_ROI_SIZE = 256
        
        # Redimensiona o ROI para o tamanho alvo.
        # cv2.INTER_LINEAR ou INTER_CUBIC são bons para ampliação (upscaling)
        zoomed_roi = cv2.resize(
            roi_frame, 
            (TARGET_ROI_SIZE, TARGET_ROI_SIZE), 
            interpolation=cv2.INTER_LINEAR
        )
        
        # O resto do processamento é feito no 'zoomed_roi'
        rgb_frame = cv2.cvtColor(zoomed_roi, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        self.hand_detected = results.multi_hand_landmarks is not None
        
        if self.hand_detected:
            hand_landmarks = results.multi_hand_landmarks[0]
            self.hand_openness = self.calculate_openness(hand_landmarks.landmark, arm_length)
            
            # Recalcula centro da mão para o frame original
            # O centro do pulso (landmark 0) é normalizado (0-1) dentro do zoomed_roi
            # Multiplicamos pela dimensão do ROI original e adicionamos o offset (x1, y1)
            wrist = hand_landmarks.landmark[0]
            self.hand_center = (
                int(wrist.x * (x2 - x1)) + x1,
                int(wrist.y * (y2 - y1)) + y1
            )
        else:
            self.hand_openness = 0.0
            self.hand_center = None
        
        return self.hand_detected, self.hand_openness, self.hand_center


# ==================== ANALISADOR DE GESTOS ====================
class GestureAnalyzer:
    def __init__(self):
        self.current_state = 'neutral' # 'open', 'closed', 'neutral'
        self.current_time = time.time() # Começa dizendo que não foi uma transiçao rapida
        self.last_state = 'neutral'  # 'open', 'closed', 'neutral'
        # self.transition_time = None  # Quando fez a transição
        self.holding_target = False  # Se está mantendo o estado alvo
        # self.target_state = None  # Estado que está mantendo
        
    def reset(self):
        """Reseta o analisador"""
        self.current_state = 'neutral'
        self.current_time = time.time()
        self.last_state = 'neutral'
        # self.transition_time = None
        self.holding_target = False
        # self.target_state = None
    
    def get_state(self, openness):
        """Retorna o estado atual baseado na abertura"""
        if openness > Config.OPEN_HAND_THRESHOLD:
            return 'open'
        elif openness < Config.CLOSED_HAND_THRESHOLD:
            return 'closed'
        else:
            return 'neutral'  # Estado neutro/intermediário
    
    def analyze(self, openness):
        """
        ## Lógica:
        1. Detecta transição rápida (< 0.3s) entre open/closed
        2. Começa a contar tempo no novo estado
        3. Se mantiver >= GESTURE_HOLD_TIME → executa comando
        ## Mas Como faz isso:
        1. Verifica se mudou de estado (open/closed/neutral)
        2. Se mudou, verifica o tempo do estado que estava (lembrando que "estava" no valor do current_state, que ainda não mudamos)
        3. Se current_time < QUICK_TRANSITION_TIME:
                Verifica se new_state é oposto do last_state → transição rápida
                Começa a contar tempo no new_state (holding_target=True)
        4. reseta state
        """
        now = time.time()     # lembre-se, tem o new e o current. 
        new_state = self.get_state(openness)
        
        # ===== Detecta mudança de estado =====
        if new_state != self.current_state:   # Mudou de estado
            # self.current_time é o momento que entrou no estado current_state
            # Condição 1: Passa pelo neutro rápido
            if now - self.current_time < Config.QUICK_TRANSITION_TIME:
                # Mudou rápido de estado
                if (self.last_state in ['open', 'closed'] and 
                    new_state in ['open', 'closed'] and 
                    self.last_state != new_state):  # Garante que não saiu pro neutro e voltou
                    # Transição rápida detectada!
                    print(f"⚡ Transição rápida: {self.last_state} → {new_state}")
                    self.holding_target = True
                else:
                    # Mudou para neutro ou mesma coisa, cancela
                    self.holding_target = False
            # Condição 2: Mudou tão rápido que nem deu tempo de registrar o neutro
            elif (  
               (self.current_state == 'open' and new_state == 'closed')
               or (self.current_state == 'closed' and new_state == 'open')
            ): 
                # Transição direta rápida
                print(f"⚡ Transição direta rápida: {self.current_state} → {new_state}")
                self.holding_target = True
  
            # Atualiza estado
            self.last_state = self.current_state
            self.current_state = new_state
            self.current_time = now
        
        # ===== Verifica se está mantendo o estado por período longo =====
        if self.holding_target:
            time_holding = now - self.current_time
            if time_holding >  Config.GESTURE_HOLD_TIME:
                # Manteve tempo suficiente, executa comando
                gesture = None
                if self.current_state == 'open':
                    gesture = 'turn_on'
                    print(f"✅ COMANDO: LIGAR LUZ (manteve aberta por {time_holding:.1f}s)")
                elif self.current_state == 'closed':
                    gesture = 'turn_off'
                    print(f"✅ COMANDO: DESLIGAR LUZ (manteve fechada por {time_holding:.1f}s)")
                
                self.reset()
                return gesture
        
        return None


# ==================== SISTEMA PRINCIPAL ====================
class GestureControlSystem:
    def __init__(self, use_gpu=True):
        self.state = SystemState.IDLE
        
        # Inicializa detectores com configuração de GPU
        self.motion_detector = MotionDetector(
            use_cuda=Config.USE_CUDA_MOG2 and use_gpu
        )
        self.pose_detector = PoseDetector(use_gpu=use_gpu)
        self.hand_detector = HandDetector(use_gpu=use_gpu)
        self.gesture_analyzer = GestureAnalyzer()
        
        self.roi = None
        self.arm_length = None  # Distância punho-cotovelo para normalização
        self.arm_length_history = deque(maxlen=10)  # Histórico para suavização
        self.cooldown_end_time = 0
        self.fps_counter = deque(maxlen=Config.FPS_AVERAGE_FRAMES)
        self.last_frame_time = time.time()
        
        # Frame skipping para performance
        self.frame_count = 0
        self.last_processed_results = None
    
    def get_stable_arm_length(self, new_arm_length):
        """
        Retorna arm_length estável usando histórico.
        Ignora valores muito pequenos ou None.
        """
        # Valida novo valor
        if new_arm_length and new_arm_length > 0.05:  # Mínimo razoável
            self.arm_length_history.append(new_arm_length)
        
        # Se temos histórico, usa média
        if len(self.arm_length_history) > 0:
            return np.median(self.arm_length_history)  # Mediana é mais robusta
        
        # Fallback: valor padrão se nunca detectamos
        return 0.15  # Valor médio típico
        
    def process_frame(self, frame):
        """Processa um frame e retorna comando (se houver)"""
        current_time = time.time()
        command = None
        
        # Calcula FPS
        self.fps_counter.append(1.0 / (current_time - self.last_frame_time))
        self.last_frame_time = current_time
        
        # Frame skipping para performance - só processa a cada N frames
        self.frame_count += 1
        should_process = (self.frame_count % Config.PROCESS_EVERY_N_FRAMES == 0)
        
        # Sempre processa em estados críticos
        if self.state in [SystemState.GESTURE_DETECTION, SystemState.ARM_RAISED]:
            should_process = True
        
        if not should_process and self.last_processed_results:
            # Reutiliza último resultado para manter fluidez visual
            return None
        
        # ===== ESTADO: IDLE =====
        if self.state == SystemState.IDLE:
            motion, _ = self.motion_detector.detect(frame)
            if motion:
                self.state = SystemState.MONITORING
                
        # ===== ESTADO: MONITORING =====
        elif self.state == SystemState.MONITORING:
            person, arm_raised, hand_pos, arm_length, _ = self.pose_detector.detect(frame)
            
            if not person:
                self.state = SystemState.IDLE
            elif arm_raised and hand_pos:
                self.state = SystemState.ARM_RAISED
                # Usa arm_length estabilizado
                self.arm_length = self.get_stable_arm_length(arm_length)
                self.roi = self.hand_detector.get_roi(frame, hand_pos, self.arm_length)
                
        # ===== ESTADO: ARM_RAISED =====
        elif self.state == SystemState.ARM_RAISED:
            person, arm_raised, hand_pos, arm_length, _ = self.pose_detector.detect(frame)
            
            if not person or not arm_raised:
                self.state = SystemState.MONITORING
                self.roi = None
            else:
                # Atualiza arm_length estabilizado
                self.arm_length = self.get_stable_arm_length(arm_length)
                self.roi = self.hand_detector.get_roi(frame, hand_pos, self.arm_length)
                
                # Detecta mão no ROI com normalização por arm_length
                hand_found, openness, _ = self.hand_detector.detect(frame, self.roi, self.arm_length)
                
                if hand_found:
                    self.state = SystemState.GESTURE_DETECTION
                    self.gesture_analyzer.reset()
                    
        # ===== ESTADO: GESTURE_DETECTION =====
        elif self.state == SystemState.GESTURE_DETECTION:
            person, arm_raised, hand_pos, arm_length, _ = self.pose_detector.detect(frame)
            
            if not person or not arm_raised:
                self.state = SystemState.MONITORING
                self.gesture_analyzer.reset()
                self.roi = None
            else:
                # Atualiza arm_length estabilizado
                self.arm_length = self.get_stable_arm_length(arm_length)
                self.roi = self.hand_detector.get_roi(frame, hand_pos, self.arm_length)
                
                # Detecta mão e analisa gesto com normalização
                hand_found, openness, _ = self.hand_detector.detect(frame, self.roi, self.arm_length)
                
                if hand_found:
                    gesture = self.gesture_analyzer.analyze(openness)
                    
                    if gesture:
                        command = gesture
                        self.state = SystemState.COOLDOWN
                        self.cooldown_end_time = current_time + Config.COOLDOWN_TIME
                        self.gesture_analyzer.reset()
                else:
                    # Perdeu a mão
                    self.state = SystemState.ARM_RAISED
                    self.gesture_analyzer.reset()
                    
        # ===== ESTADO: COOLDOWN =====
        elif self.state == SystemState.COOLDOWN:
            if current_time >= self.cooldown_end_time:
                self.state = SystemState.MONITORING
                self.roi = None
                self.arm_length = None
        
        return command
    
    def draw_debug(self, frame):
        """Desenha informações de debug no frame"""
        if not Config.SHOW_DEBUG:
            return frame
        
        h, w = frame.shape[:2]
        debug_frame = frame.copy()
        
        # FPS
        fps = np.mean(self.fps_counter) if self.fps_counter else 0
        fps_color = (0, 255, 0) if fps > 20 else (0, 165, 255) if fps > 12 else (0, 0, 255)
        cv2.putText(debug_frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, fps_color, 2)
        
        # Configurações de performance
        perf_text = f"Res:{Config.FRAME_WIDTH}x{Config.FRAME_HEIGHT} Skip:{Config.PROCESS_EVERY_N_FRAMES} Model:Lite"
        cv2.putText(debug_frame, perf_text, (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # GPU Status
        gpu_status = "GPU (render only)" if Config.USE_GPU else "CPU"
        cv2.putText(debug_frame, f"Mode: {gpu_status}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # Estado
        state_colors = {
            SystemState.IDLE: (128, 128, 128),
            SystemState.MONITORING: (255, 255, 0),
            SystemState.ARM_RAISED: (0, 255, 255),
            SystemState.GESTURE_DETECTION: (0, 255, 0),
            SystemState.COOLDOWN: (255, 0, 255)
        }
        color = state_colors[self.state]
        cv2.putText(debug_frame, f"Estado: {self.state.name}", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # ROI - CORRIGIDO para aparecer sempre que existir
        if self.roi is not None:
            x1, y1, x2, y2 = self.roi
            roi_color = (0, 255, 255) if self.state == SystemState.ARM_RAISED else (0, 255, 0)
            cv2.rectangle(debug_frame, (x1, y1), (x2, y2), roi_color, 2)
            roi_size = x2 - x1
            cv2.putText(debug_frame, f"ROI: {roi_size}px", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, roi_color, 1)
        
        # Hand openness - MELHORADO com indicador visual
        if self.state in [SystemState.GESTURE_DETECTION, SystemState.ARM_RAISED]:
            openness = self.hand_detector.hand_openness
            
            # Mostra arm_length para debug
            if self.arm_length:
                arm_len_text = f"Escala: {self.arm_length:.3f}"
                if len(self.arm_length_history) > 0:
                    arm_len_text += f" (hist:{len(self.arm_length_history)})"
                cv2.putText(debug_frame, arm_len_text, (10, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            
            # Texto com cor baseada no valor
            if openness > Config.OPEN_HAND_THRESHOLD:
                openness_color = (0, 255, 0)  # Verde = ABERTA
                state_text = "ABERTA"
            elif openness < Config.CLOSED_HAND_THRESHOLD:
                openness_color = (0, 0, 255)  # Vermelho = FECHADA
                state_text = "FECHADA"
            else:
                openness_color = (0, 165, 255)  # Laranja = INTERMEDIÁRIA
                state_text = "INTERMEDIARIA"
            
            cv2.putText(debug_frame, f"Abertura: {openness:.2f} ({state_text})", (10, 165),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, openness_color, 2)
            
            # Modo calibração
            if Config.CALIBRATION_MODE:
                cv2.putText(debug_frame, "MODO CALIBRACAO ATIVO", (10, 190),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.putText(debug_frame, "(veja valores [CALIB] no terminal)", (10, 210),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            
            # Barra visual de abertura
            bar_x = 10
            bar_y = 220
            bar_width = 200
            bar_height = 20
            
            cv2.rectangle(debug_frame, (bar_x, bar_y), 
                        (bar_x + bar_width, bar_y + bar_height),
                        (100, 100, 100), -1)
            
            fill_width = int(bar_width * openness)
            cv2.rectangle(debug_frame, (bar_x, bar_y),
                        (bar_x + fill_width, bar_y + bar_height),
                        openness_color, -1)
            
            # Linhas dos thresholds
            closed_pos = int(bar_width * Config.CLOSED_HAND_THRESHOLD)
            open_pos = int(bar_width * Config.OPEN_HAND_THRESHOLD)
            
            cv2.line(debug_frame, (bar_x + closed_pos, bar_y),
                    (bar_x + closed_pos, bar_y + bar_height),
                    (0, 0, 255), 2)
            
            cv2.line(debug_frame, (bar_x + open_pos, bar_y),
                    (bar_x + open_pos, bar_y + bar_height),
                    (0, 255, 0), 2)
            
            # ===== BARRA DE PROGRESSO DO GESTO (LÓGICA CORRIGIDA) =====
            if self.gesture_analyzer.holding_target:
                progress = 0
                gesture_type = ""
                if self.gesture_analyzer.current_state == 'open':
                    gesture_type = "Mantenha ABERTA para LIGAR"
                elif self.gesture_analyzer.current_state == 'closed':
                    gesture_type = "Mantenha FECHADA para DESLIGAR"

                if self.gesture_analyzer.current_time:
                    elapsed = time.time() - self.gesture_analyzer.current_time
                    progress = min(max(elapsed / Config.GESTURE_HOLD_TIME, 0.0), 1.0)
                else:
                    progress = 0

                prog_bar_width = 300
                prog_bar_height = 30
                prog_bar_x = w - prog_bar_width - 20
                prog_bar_y = 20

                # Texto do gesto
                cv2.putText(debug_frame, gesture_type, (prog_bar_x, prog_bar_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # Barra de progresso
                cv2.rectangle(debug_frame, (prog_bar_x, prog_bar_y),
                              (prog_bar_x + prog_bar_width, prog_bar_y + prog_bar_height),
                              (255, 255, 255), 2)
                cv2.rectangle(debug_frame, (prog_bar_x, prog_bar_y),
                              (prog_bar_x + int(prog_bar_width * progress), prog_bar_y + prog_bar_height),
                              (0, 255, 0), -1)

                # Texto de progresso
                cv2.putText(debug_frame, f"{progress*100:.0f}%",
                            (prog_bar_x + prog_bar_width//2 - 20, prog_bar_y + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            # # Estado atual da mão (quando não está fazendo gesto)
            # if not self.gesture_analyzer.holding_target:
            #     state_text = f"Estado: {self.gesture_analyzer.get_state(self.hand_detector.hand_openness)}"
            #     if self.gesture_analyzer.state_start_time:
            #         time_in_state = time.time() - self.gesture_analyzer.state_start_time
            #         state_text += f" ({time_in_state:.1f}s)"
            #         # Mostra quando está pronto para fazer gesto
            #         if time_in_state >= Config.IDLE_RESET_TIME:
            #             state_text += " - PRONTO!"
            #             text_color = (0, 255, 0)
            #         else:
            #             text_color = (200, 200, 200)
            #     else:
            #         text_color = (200, 200, 200)
            #     cv2.putText(debug_frame, state_text, (w - 350, 30),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
        
        # Instruções
        instructions = [
            "Pressione 'q' para sair | 'c' para calibracao | 'd' para debug",
            "1. Levante o braco",
            "2. Mantenha mao aberta/fechada ate aparecer 'PRONTO'",
            "3. Mude rapido para outro estado e mantenha 1s"
        ]
        
        y_pos = h - 90
        for instruction in instructions:
            cv2.putText(debug_frame, instruction, (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            y_pos += 20
        
        return debug_frame


# ==================== FUNÇÃO PRINCIPAL ====================
def main():
    # Mostra informações de GPU
    GPUInfo.print_gpu_info()
    
    # Função para enviar comandos ao Home Assistant
    def send_light_command(command):
        """Envia comando para o Home Assistant"""
        print(f"🔥 COMANDO ENVIADO: {command}")
        try:
            if command == 'turn_on':
                ha_libs.turn_on_light("light.0xa4c138254b8958b5", brightness=255, color_temp=153, hs_color=(100, 100))
                print("✅ Luz LIGADA com sucesso!")
            elif command == 'turn_off':
                ha_libs.turn_off_light("light.0xa4c138254b8958b5")
                print("✅ Luz DESLIGADA com sucesso!")
        except Exception as e:
            print(f"❌ Erro ao comunicar com Home Assistant: {e}")
    
    # Inicializa sistema com GPU
    system = GestureControlSystem(use_gpu=Config.USE_GPU)
    cap = cv2.VideoCapture(Config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)
    
    print("\n🎮 Sistema iniciado!")
    print("Pressione 'q' para sair")
    print("Pressione 'c' para ativar/desativar modo calibração")
    print("Pressione 'd' para ativar/desativar debug visual\n")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Processa frame
            command = system.process_frame(frame)
            
            # Executa comando se houver
            if command:
                send_light_command(command)
            
            # Mostra frame com debug
            debug_frame = system.draw_debug(frame)
            cv2.imshow('Gesture Control System', debug_frame)
            
            # Teclas de controle
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                Config.CALIBRATION_MODE = not Config.CALIBRATION_MODE
                status = "ATIVADO" if Config.CALIBRATION_MODE else "DESATIVADO"
                print(f"\n🔧 Modo Calibração: {status}")
                if Config.CALIBRATION_MODE:
                    print("Faça gestos e observe os valores [CALIB] no terminal")
            elif key == ord('d'):
                Config.SHOW_DEBUG = not Config.SHOW_DEBUG
                status = "ATIVADO" if Config.SHOW_DEBUG else "DESATIVADO"
                print(f"\n👁️  Debug Visual: {status}")
                
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()