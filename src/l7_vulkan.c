/* l7_vulkan.c ? backend GPU fase 1: sonda de dispositivos (carga dinamica, sin SDK) */
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan_core.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
#endif
#include "g2b.h"

static HMODULE g_dll; static VkInstance g_inst; static VkPhysicalDevice g_pd;
static VkDevice g_dev; static u32 g_qfam; static char g_name[256]; static u64 g_vram;

static PFN_vkGetInstanceProcAddr p_GIPA;
static PFN_vkCreateInstance p_vkCreateInstance;
static PFN_vkEnumeratePhysicalDevices p_vkEnumeratePhysicalDevices;
static PFN_vkGetPhysicalDeviceProperties p_vkGetPhysicalDeviceProperties;
static PFN_vkGetPhysicalDeviceMemoryProperties p_vkGetPhysicalDeviceMemoryProperties;
static PFN_vkEnumerateDeviceExtensionProperties p_vkEnumerateDeviceExtensionProperties;
static PFN_vkCreateDevice p_vkCreateDevice;
static PFN_vkGetDeviceQueue p_vkGetDeviceQueue;
static PFN_vkDestroyInstance p_vkDestroyInstance;
static PFN_vkDestroyDevice p_vkDestroyDevice;

static FILE *vlog(void){ return fopen("vk.log","a"); }
int vk_init(void){
#ifdef _WIN32
  fprintf(stderr,"[vk] cargando vulkan-1.dll...\n");
  { FILE *L=vlog(); fprintf(L,"1: cargando dll\n"); fclose(L); }
  g_dll=LoadLibraryA("vulkan-1.dll");
#else
  g_dll=dlopen("libvulkan.so.1",RTLD_NOW);
#endif
  if(!g_dll){ fprintf(stderr,"vk: runtime no encontrado\n"); return -1; }
#ifdef _WIN32
  p_GIPA=(PFN_vkGetInstanceProcAddr)(void(*)(void))GetProcAddress(g_dll,"vkGetInstanceProcAddr");
#else
  p_GIPA=(PFN_vkGetInstanceProcAddr)dlsym(g_dll,"vkGetInstanceProcAddr");
#endif
  { FILE *L=vlog(); fprintf(L,"2: dll cargada\n"); fclose(L); }
  if(!p_GIPA){ fprintf(stderr,"vk: sin vkGetInstanceProcAddr\n"); return -1; }
#define GETI(x) p_##x=(PFN_##x)p_GIPA(VK_NULL_HANDLE,#x)
  GETI(vkCreateInstance); GETI(vkEnumeratePhysicalDevices); GETI(vkGetPhysicalDeviceProperties);
  GETI(vkGetPhysicalDeviceMemoryProperties); GETI(vkCreateDevice); GETI(vkGetDeviceQueue);
  GETI(vkDestroyInstance); GETI(vkDestroyDevice);
#undef GETI
#define GETI(x) p_##x=(PFN_##x)p_GIPA(VK_NULL_HANDLE,#x)
  GETI(vkCreateInstance); GETI(vkEnumeratePhysicalDevices); GETI(vkGetPhysicalDeviceProperties);
  GETI(vkGetPhysicalDeviceMemoryProperties); GETI(vkEnumerateDeviceExtensionProperties);
  GETI(vkCreateDevice); GETI(vkGetDeviceQueue); GETI(vkDestroyInstance); GETI(vkDestroyDevice);
#undef GETI
  if(!p_vkCreateInstance){ fprintf(stderr,"vk: sin vkCreateInstance\n"); return -1; }
  VkApplicationInfo ai; memset(&ai,0,sizeof ai);
  ai.sType=VK_STRUCTURE_TYPE_APPLICATION_INFO;
  ai.pApplicationName="gguf2bin2"; ai.applicationVersion=VK_MAKE_VERSION(4,5,0);
  ai.apiVersion=VK_MAKE_API_VERSION(0,1,0,0);
  { FILE *L=vlog(); fprintf(L,"3: punteros listos\n"); fclose(L); }
  VkInstanceCreateInfo ci; memset(&ci,0,sizeof ci);
  ci.sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO; ci.pApplicationInfo=&ai;
  fprintf(stderr,"[vk] creando instancia (api 1.1)...\n");
  if(p_vkCreateInstance(&ci,NULL,&g_inst)!=VK_SUCCESS){ fprintf(stderr,"vk: fallo instancia\n"); return -1; }
  fprintf(stderr,"[vk] instancia OK, enumerando GPUs...\n");
#define GETP(x) (PFN_##x)p_GIPA((VkInstance)g_inst,#x)
  PFN_vkEnumeratePhysicalDevices ePD=GETP(vkEnumeratePhysicalDevices);
  u32 n=0; if(p_vkEnumeratePhysicalDevices(g_inst,&n,NULL)||!n){ fprintf(stderr,"vk: 0 GPUs\n"); return -1; }
  VkPhysicalDevice *pds=malloc(n*sizeof(void*));
  p_vkEnumeratePhysicalDevices(g_inst,&n,pds);
  for(u32 i=0;i<n;i++){
    VkPhysicalDeviceProperties pr; p_vkGetPhysicalDeviceProperties(pds[i],&pr);
    printf("vk[%u]: %s (vendor %04x id %04x tipo %u)\n",i,pr.deviceName,pr.vendorID,pr.deviceID,pr.deviceType);
    VkPhysicalDeviceMemoryProperties mp; p_vkGetPhysicalDeviceMemoryProperties(pds[i],&mp);
    for(u32 h=0;h<mp.memoryHeapCount;h++)
      printf("   heap %u: %llu MB%s\n",h,(unsigned long long)(mp.memoryHeaps[h].size>>20),
        (mp.memoryHeaps[h].flags&VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)?" [device-local]":"");
  }
  /* elegir discreta si existe; si no, la primera */
  g_pd=pds[0];
  for(u32 i=0;i<n;i++){ VkPhysicalDeviceProperties pr; p_vkGetPhysicalDeviceProperties(pds[i],&pr);
    if(pr.deviceType==2/*DISCRETE*/){ g_pd=pds[i]; break; } }
  { VkPhysicalDeviceProperties pr; p_vkGetPhysicalDeviceProperties(g_pd,&pr);
    strncpy(g_name,pr.deviceName,255); g_name[255]=0; }
  { VkPhysicalDeviceMemoryProperties mp; p_vkGetPhysicalDeviceMemoryProperties(g_pd,&mp);
    for(u32 h=0;h<mp.memoryHeapCount;h++)
      if(mp.memoryHeaps[h].flags&VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) g_vram=mp.memoryHeaps[h].size; }
  free(pds);
  return 0;
}
void vk_shutdown(void){ if(g_dev&&p_vkDestroyDevice) p_vkDestroyDevice(g_dev,NULL); if(g_inst&&p_vkDestroyInstance) p_vkDestroyInstance(g_inst,NULL); g_dev=NULL; g_inst=NULL; }
const char *vk_device_name(void){ return g_name; }
u64 vk_vram(void){ return g_vram; }
