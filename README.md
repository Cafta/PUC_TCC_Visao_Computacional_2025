## Home Vision — Controle por Gestos

Resumo rápido
--------------
Este projeto implementa um sistema de controle por gestos usando OpenCV e MediaPipe. Ele detecta movimento, identifica pessoa e posição do braço (braço levantado), recorta uma ROI ao redor da mão, estima o "nível de abertura" da mão (aberta/fechada) e reconhece gestos simples para controlar uma entidade do Home Assistant (ex.: ligar/desligar luz).

Principais componentes
----------------------
- `home_vision/home_vision_v7.4.6.py`: script principal que contém toda a lógica (configurações, detectores de movimento/pose/mão, análise de gesto e loop de vídeo).
- `home_vision/libs/home_assistant_lib.py`: biblioteca responsável pela comunicação com o Home Assistant (invocada quando um gesto valida um comando).

Funcionalidades
---------------
- Detecção de movimento por background subtraction (MOG2) com opção de usar CUDA (se compilado/ disponível).
- Detecção de pose com MediaPipe para identificar quando o usuário levanta o braço.
- Detecção de mão dentro de uma ROI dinamicamente calculada; zoom do ROI para melhorar precisão.
- Cálculo de "hand openness" (0.0 a 1.0) com normalização pela escala do braço, usando múltiplas métricas (distância das pontas ao pulso, spread, extensão).
- Lógica de reconhecimento de gesto baseada em transições rápidas e tempo de manutenção (ex.: manter mão aberta 1s → ligar luz).
- Integração com Home Assistant: comandos `turn_on` / `turn_off` são enviados via `home_assistant_lib`.
- Modo de calibração (tecla 'c') e debug visual (tecla 'd').

Requisitos
----------
- Python 3.8+ (testado com versões recentes de CPython)
- Pacotes (veja `requirements.txt`):
  - opencv-python (ou OpenCV compilado com CUDA para aceleração opcional)
  - mediapipe
  - numpy

Instalação rápida
-----------------
1. Crie um ambiente virtual (recomendado).
2. Instale dependências:

```powershell
pip install -r requirements.txt
```

Uso
---
1. Ajuste as configurações no topo de `home_vision/home_vision_v7.4.6.py` (classe `Config`): índice da câmera (`CAMERA_INDEX`), resolução (`FRAME_WIDTH`, `FRAME_HEIGHT`), thresholds de gesto, e flags de GPU/Calibração.
2. Execute o script:

```powershell
python .\home_vision\home_vision_v7.4.6.py
```

Teclas úteis durante a execução
- q : sair
- c : alternar modo calibração (mostra logs [CALIB] no terminal)
- d : alternar debug visual (overlay sobre o vídeo)

Como funciona (alto nível)
-------------------------
1. Background subtraction (MOG2) detecta movimento para sair do estado IDLE.
2. MediaPipe Pose detecta se há uma pessoa e se o braço está levantado.
3. Se braço levantado, calcula ROI ao redor da mão e aplica MediaPipe Hands em um ROI ampliado (zoom) para melhorar precisão.
4. Abertura da mão é medida e analisada pelo `GestureAnalyzer` que detecta transições e confirma gestos por tempo de manutenção.
5. Comandos são enviados ao Home Assistant via `home_assistant_lib`.

Integração com Home Assistant
-----------------------------
O envio de comandos está implementado na função `send_light_command` dentro do script principal. O código chama:

- `ha_libs.turn_on_light("light.0xa4c138254b8958b5", brightness=5)`
- `ha_libs.turn_off_light("light.0xa4c138254b8958b5")`

Substitua o `entity_id` pelo seu próprio identificador de dispositivo no Home Assistant.

Observações sobre GPU
---------------------
- MediaPipe Python normalmente roda a inferência em CPU (TensorFlow Lite). A aceleração GPU real para inferência não está plenamente disponível via MediaPipe Python API — o script faz checagens informativas e usa OpenCV+CUDA apenas para MOG2 se disponível.
- Para ganhos de performance, reduza resolução ou aumente `PROCESS_EVERY_N_FRAMES`.

Dicas e resolução de problemas
------------------------------
- Se não houver vídeo: verifique `CAMERA_INDEX` (pode ser 0, caminho RTSP/HTTP, ou dispositivo USB).
- Se a comunicação com o Home Assistant falhar: verifique `home_vision/libs/home_assistant_lib.py` e as credenciais/configuração.
- Para melhorar detecção em ambientes fracos: ajuste `MOG2_VAR_THRESHOLD`, `MIN_CONTOUR_AREA` e thresholds de abertura de mão.

Limitações e casos conhecidos
----------------------------
- Projeta detecção para 1 pessoa / 1 mão (configurado `max_num_hands=1`).
- O cálculo de abertura da mão foi calibrado empiricamente; resultados variam com câmera, distância e iluminação.

Próximos passos sugeridos
------------------------
- Extrair configurações para um arquivo `.yaml`/`.json` para facilitar ajustes sem editar código.
- Adicionar testes unitários para `calculate_openness` e ROI.
- Melhorar integração com Home Assistant (autenticação, retries, logs).

Arquivos de interesse
---------------------
- `home_vision/home_vision_v7.4.6.py` — script principal
- `home_vision/libs/home_assistant_lib.py` — integração com Home Assistant

Contato / autoria
-----------------
Arquivo original no repositório. Não há metadados explícitos de autoria/licença no código; adicione um `LICENSE` se desejar publicar.

---
Gerado automaticamente a partir do código fonte `home_vision_v7.4.6.py` para fornecer visão geral e instruções básicas.
