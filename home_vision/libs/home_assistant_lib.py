import requests
import json

# ===== CONFIGURAÇÕES =====
# Substitua pelo seu token gerado no Home Assistant
TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiI1YzU0YzIzNzAyZmU0MTVhYjdjYWMyMjBiMDg0MGM4NiIsImlhdCI6MTc2MDI3Mzc2MCwiZXhwIjoyMDc1NjMzNzYwfQ.NWzVNV-85nsG0_q8yG7pExyZqIXMvL9z3rkzx1rFaQM"

# URL do seu Home Assistant (ajuste a porta se necessário)
BASE_URL = "http://localhost:8123"

# ===== FUNÇÃO PARA BUSCAR ENTIDADES =====
def get_entities():
    """
    Busca todas as entidades do Home Assistant
    """
    # Monta a URL completa
    url = f"{BASE_URL}/api/states"
    
    # Configura o header com o token de autenticação
    headers = {
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": "application/json"
    }
    
    try:
        # Faz a requisição GET
        response = requests.get(url, headers=headers)
        
        # Verifica se deu certo (código 200)
        if response.status_code == 200:
            # Converte a resposta JSON em objeto Python
            entities = response.json()
            return entities
        else:
            print(f"Erro: Status code {response.status_code}")
            print(f"Mensagem: {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"Erro na conexão: {e}")
        return None

def show_entities():
    entities = get_entities()

    if entities:
        print(f"\nTotal de entidades encontradas: {len(entities)}\n")
        
        # Mostra cada entidade
        for entity in entities:
            print(f"ID: {entity['entity_id']}")
            print(f"  Estado: {entity['state']}")
            print(f"  Nome: {entity['attributes'].get('friendly_name', 'N/A')}")
            print(f"  Atributos: {list(entity['attributes'].keys())}")
            print("-" * 50)
    else:
        print("Não foi possível buscar as entidades.")

def turn_on_light(entity_id, brightness=None, color_temp=None, hs_color=None, rgb_color=None):
    """
    Liga uma luz no Home Assistant
    
    Args:
        entity_id: ID da entidade (ex: 'light.abajour')
        brightness: Brilho de 0 a 255 (opcional)
        color_temp: Temperatura de 153 a 500 (opcional)
        hs_color: Tupla com cor HS [hue, saturation] (opcional)
        rgb_color: Tupla com cor RGB [Red, Green, Blue] (opcional - prioritario)
    """
    # Monta a URL para o serviço light.turn_on
    url = f"{BASE_URL}/api/services/light/turn_on"
    
    # Header com autenticação
    headers = {
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": "application/json"
    }
    
    # Monta o payload (dados a enviar)
    payload = {
        "entity_id": entity_id
    }

    # Adiciona brilho se foi especificado
    if brightness is not None:
        orig = brightness
        brightness = max(0, min(brightness, 255))
        if brightness != orig:
            print(f"Brilho = {orig}? Tem que ser entre 0 e 255 → alterado para {brightness}")
        payload["brightness"] = brightness
    
    # Adiciona temperatura se foi especificado
    if color_temp is not None:
        orig = color_temp
        color_temp = max(153, min(color_temp, 500))
        if color_temp != orig:
            print(f"Temperatura (color_temp) = {orig}? Tem que ser entre 153 e 500 → alterado para {color_temp}")  
        payload["color_temp"] = color_temp
    # Adiciona cor se foi especificada (prioridade para a temperatura - não dá para colocar os dois)
    elif rgb_color is not None and len(rgb_color) == 3:
        orig = rgb_color
        rgb_color = tuple(max(0, min(v, 255)) for v in rgb_color    )
        if rgb_color != orig:
            print(f"Cor (RGB) = {orig}? As cores tem que estar entre 0 e 255 → alterado para {rgb_color}")  
        payload["rgb_color"] = rgb_color
    elif hs_color is not None and len(hs_color) == 2:
        orig = hs_color
        hs_color = tuple(max(0, min(v, 100)) for v in hs_color    )
        if rgb_color != orig:
            print(f"Cor (hs) = {orig}? As cores tem que estar entre 0 e 100 → alterado para {hs_color}") 
        payload["hs_color"] = hs_color
    
    try:
        # Faz a requisição POST
        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code == 200:
            print(f"✓ Luz ligada com sucesso!")
            print(f"  Entity: {entity_id}")
            if brightness:
                print(f"  Brilho: {brightness}/255")
            if color_temp:
                print(f"  Temperatura (color_temp): {color_temp}")
            if hs_color:
                if color_temp:
                    print("  Cor (HS) ignorada, não pode ter temperatura e cor no mesmo comando")
                else: print(f"  Cor (HS): {hs_color}")
            if rgb_color:
                if color_temp:
                    print("  Cor (RGB) ignorada, não pode ter temperatura e cor no mesmo comando")
                else: print(f"  Cor (RGB): {rgb_color}")
            return True
        else:
            print(f"✗ Erro: Status code {response.status_code}")
            print(f"  Mensagem: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"✗ Erro na conexão: {e}")
        return False


# ===== FUNÇÃO PARA DESLIGAR LUZ =====
def turn_off_light(entity_id):
    """
    Desliga uma luz no Home Assistant
    """
    url = f"{BASE_URL}/api/services/light/turn_off"
    
    headers = {
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "entity_id": entity_id
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code == 200:
            print(f"✓ Luz desligada com sucesso!")
            print(f"  Entity: {entity_id}")
            return True
        else:
            print(f"✗ Erro: Status code {response.status_code}")
            print(f"  Mensagem: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"✗ Erro na conexão: {e}")
        return False


# ===== PROGRAMA PRINCIPAL =====
if __name__ == "__main__":
    LED_ID = "light.0xa4c138254b8958b5"

    print("Buscando entidades do Home Assistant...")
    print("=" * 50)
    

    # show_entities()
    turn_off_light(LED_ID)
    #turn_on_light(LED_ID, brightness=255, color_temp=153, hs_color=(100, 100))
    
