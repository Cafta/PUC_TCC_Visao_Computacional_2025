#ifndef HOME_ASSISTENT_LIB_H
#define HOME_ASSISTENT_LIB_H

#include "chaves.h"

// Busca todas as entidades do Home Assistant
void* get_entities(void);

// Mostra todas as entidades do Home Assistant
void show_entities(void);

// Liga uma luz no Home Assistant
int turn_on_light(const char* entity_id, int brightness, int color_temp, int* hs_color, int* rgb_color);

// Desliga uma luz no Home Assistant
int turn_off_light(const char* entity_id);

#endif // HOME_ASSISTENT_LIB_H
