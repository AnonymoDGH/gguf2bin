/* version.h - fuente unica de verdad para la version de gguf2bin2.
   Todo codigo que imprima o compare versiones usa estos defines. */
#ifndef G2B_VERSION_H
#define G2B_VERSION_H

#define G2B_VERSION_MAJOR 4
#define G2B_VERSION_MINOR 9
#define G2B_VERSION_PATCH 0

#define G2B_STR_(x) #x
#define G2B_STR(x) G2B_STR_(x)
#define G2B_VERSION G2B_STR(G2B_VERSION_MAJOR) "." G2B_STR(G2B_VERSION_MINOR) "." G2B_STR(G2B_VERSION_PATCH)

#endif
