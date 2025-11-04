# 🏠 Guia Rápido: Iniciar Sistema Smart Home
```
📁 Estrutura de Pastas
~/docker/
├── homeassistant/
│   └── (configurações do HA)
└── zigbee2mqtt/
    └── data/
        └── configuration.yaml
```
Para rodar com minha placa de vídeo:  
__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia python home_vision/home_vision_v7.3.py  

# 🚀 Iniciar Serviços:   
1️⃣ Iniciar Docker (se necessário)
```
sudo systemctl start docker
sudo systemctl status docker
```
2️⃣ Iniciar MQTT Broker
```
sudo systemctl start mosquitto
sudo systemctl status mosquitto
```
3️⃣ Iniciar Home Assistant
```
sudo docker start homeassistant
# Ver logs
sudo docker logs -f homeassistant
```
4️⃣ Iniciar Zigbee2MQTT
```
sudo docker start zigbee2mqtt
# Ver logs
sudo docker logs -f zigbee2mqtt
```
✅ Verificar Status de Tudo
```
# Ver containers rodando
sudo docker ps

# Ver status do MQTT
sudo systemctl status mosquitto

# Testar MQTT
mosquitto_pub -h localhost -t test/topic -m "OK"
```

🌐 Acessar Interfaces
```
Home Assistant: http://localhost:8123
Zigbee2MQTT: http://localhost:8080
```

🔄 Reiniciar Serviços
```
# Reiniciar Home Assistant
sudo docker restart homeassistant

# Reiniciar Zigbee2MQTT
sudo docker restart zigbee2mqtt

# Reiniciar MQTT
sudo systemctl restart mosquitto
```
🛑 Parar Serviços
```
# Parar Home Assistant
sudo docker stop homeassistant

# Parar Zigbee2MQTT
sudo docker stop zigbee2mqtt

# Parar MQTT
sudo systemctl stop mosquitto
```
🔧 Comandos Úteis
```
Ver logs em tempo real
bashsudo docker logs -f homeassistant
sudo docker logs -f zigbee2mqtt
sudo journalctl -u mosquitto -f
Editar configurações
bash# Zigbee2MQTT
nano ~/docker/zigbee2mqtt/data/configuration.yaml

# Depois de editar, reiniciar:
sudo docker restart zigbee2mqtt
Testar MQTT
# Terminal 1 - escutar
mosquitto_sub -h localhost -t '#' -v

# Terminal 2 - enviar
mosquitto_pub -h localhost -t test/topic -m "Teste"
```