/* l7_vulkan.c — backend GPU fase 1+2: sonda + head GEMV Q4_0S en GPU */
#define VK_NO_PROTOTYPES
#define VK_NULL_HANDLE ((void*)0)
#include <vulkan/vulkan_core.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
#endif
#include "g2b.h"

static HMODULE g_dll;
static VkInstance g_inst; static VkPhysicalDevice g_pd;
static VkDevice g_dev; static u32 g_qfam=99; static VkQueue g_queue;
static char g_name[256]; static u64 g_vram;

/* punteros: convencion unica p_<NombreFuncionVulkan> generada por GETI/GETD */
static PFN_vkGetInstanceProcAddr p_vkGetInstanceProcAddr;
static PFN_vkCreateInstance p_vkCreateInstance;
static PFN_vkEnumeratePhysicalDevices p_vkEnumeratePhysicalDevices;
static PFN_vkGetPhysicalDeviceProperties p_vkGetPhysicalDeviceProperties;
static PFN_vkGetPhysicalDeviceMemoryProperties p_vkGetPhysicalDeviceMemoryProperties;
static PFN_vkCreateDevice p_vkCreateDevice;
static PFN_vkGetDeviceQueue p_vkGetDeviceQueue;
static PFN_vkDestroyInstance p_vkDestroyInstance;
static PFN_vkDestroyDevice p_vkDestroyDevice;
static PFN_vkGetDeviceProcAddr p_GetDeviceProcAddr;
static PFN_vkCreateBuffer p_vkCreateBuffer;
static PFN_vkAllocateMemory p_vkAllocateMemory;
static PFN_vkBindBufferMemory p_vkBindBufferMemory;
static PFN_vkMapMemory p_vkMapMemory;
static PFN_vkUnmapMemory p_vkUnmapMemory;
static PFN_vkCreateShaderModule p_vkCreateShaderModule;
static PFN_vkCreateDescriptorSetLayout p_vkCreateDescriptorSetLayout;
static PFN_vkCreatePipelineLayout p_vkCreatePipelineLayout;
static PFN_vkCreateComputePipelines p_vkCreateComputePipelines;
static PFN_vkCreateDescriptorPool p_vkCreateDescriptorPool;
static PFN_vkAllocateDescriptorSets p_vkAllocateDescriptorSets;
static PFN_vkUpdateDescriptorSets p_vkUpdateDescriptorSets;
static PFN_vkCreateCommandPool p_vkCreateCommandPool;
static PFN_vkAllocateCommandBuffers p_vkAllocateCommandBuffers;
static PFN_vkResetCommandPool p_vkResetCommandPool;
static PFN_vkBeginCommandBuffer p_vkBeginCommandBuffer;
static PFN_vkCmdBindPipeline p_vkCmdBindPipeline;
static PFN_vkCmdBindDescriptorSets p_vkCmdBindDescriptorSets;
static PFN_vkCmdPushConstants p_vkCmdPushConstants;
static PFN_vkCmdDispatch p_vkCmdDispatch;
static PFN_vkEndCommandBuffer p_vkEndCommandBuffer;
static PFN_vkQueueSubmit p_vkQueueSubmit;
static PFN_vkQueueWaitIdle p_vkQueueWaitIdle;

#define GETF(x) p_##x=(PFN_##x)p_vkGetInstanceProcAddr((VkInstance)g_inst,#x)
#define GETD(x) p_##x=(PFN_##x)p_GetDeviceProcAddr(g_dev,#x)

/* fase 2 globals */
static VkBuffer g_wbuf,g_xbuf,g_obuf;
static VkDeviceMemory g_wmem,g_xmem,g_omem;
static VkDescriptorPool g_dpool; static VkDescriptorSetLayout g_dlayout;
static VkDescriptorSet g_dset; static VkPipelineLayout g_playout; static VkPipeline g_pipe;
static VkCommandPool g_cpool; static VkCommandBuffer g_cb;
static u32 g_rows,g_nsb; static i32 g_n_head;
static u64 g_wbytes; static u32 g_rows_total,g_row_u32s;
static f32 *g_xmap,*g_omap; static int g_head_ok,g_pipe_ok;
static void head_free(void);

int vk_init(void){
#ifdef _WIN32
  { FILE *L=fopen("vk.log","a"); fprintf(L,"A: entrada\n"); fclose(L); }
  if(!getenv("VK_G2B_ALLOW_AMD") && !getenv("VK_LOADER_DRIVERS_DISABLE")) _putenv("VK_LOADER_DRIVERS_DISABLE=*amd*");
  /* Bypass del loader: este sistema no tiene HKLM\Khronos\Vulkan\Drivers y la
     enumeracion del loader crashea. Cargamos el ICD del DriverStore DIRECTAMENTE. */
  g_dll=NULL;
  {
    const char *dlls[2]={"igvk64.dll","amdvlk64.dll"};   /* AMD solo si VK_G2B_ALLOW_AMD */
    int npat=getenv("VK_G2B_ALLOW_AMD")?2:1;
    WIN32_FIND_DATAA fd;
    HANDLE h=FindFirstFileA("C:\\Windows\\System32\\DriverStore\\FileRepository\\*",&fd);
    if(h!=INVALID_HANDLE_VALUE){
      do{
        if(!(fd.dwFileAttributes&FILE_ATTRIBUTE_DIRECTORY)) continue;
        for(int di=0;di<npat&&!g_dll;di++){
          char full[MAX_PATH];
          snprintf(full,sizeof full,"C:\\Windows\\System32\\DriverStore\\FileRepository\\%s\\%s",fd.cFileName,dlls[di]);
          g_dll=LoadLibraryA(full);
          if(g_dll){ FILE *L=fopen("vk.log","a"); fprintf(L,"A2: icd directo %s\n",full); fclose(L); }
        }
      }while(!g_dll&&FindNextFileA(h,&fd));
      FindClose(h);
    }
  }
  if(!g_dll) g_dll=LoadLibraryA("vulkan-1.dll");
#else
  g_dll=dlopen("libvulkan.so.1",RTLD_NOW);
#endif
  { FILE *L=fopen("vk.log","a"); fprintf(L,"B: dll=%p\n",(void*)g_dll); fclose(L); }
  if(!g_dll){ fprintf(stderr,"vk: runtime no encontrado\n"); return -1; }
#ifdef _WIN32
  p_vkGetInstanceProcAddr=(PFN_vkGetInstanceProcAddr)(void(*)(void))GetProcAddress(g_dll,"vkGetInstanceProcAddr");
#else
  p_vkGetInstanceProcAddr=(PFN_vkGetInstanceProcAddr)dlsym(g_dll,"vkGetInstanceProcAddr");
#endif
  if(!p_vkGetInstanceProcAddr){ fprintf(stderr,"vk: sin vkGetInstanceProcAddr\n"); return -1; }
  GETF(vkCreateInstance); GETF(vkEnumeratePhysicalDevices); GETF(vkGetPhysicalDeviceProperties);
  GETF(vkGetPhysicalDeviceMemoryProperties); GETF(vkCreateDevice); GETF(vkGetDeviceQueue);
  GETF(vkDestroyInstance); GETF(vkDestroyDevice); 
  { FILE *L=fopen("vk.log","a"); fprintf(L,"C: pre-instancia\n"); fclose(L); }
  if(!p_vkCreateInstance){ fprintf(stderr,"vk: sin vkCreateInstance\n"); return -1; }
  VkApplicationInfo ai; memset(&ai,0,sizeof ai);
  ai.sType=VK_STRUCTURE_TYPE_APPLICATION_INFO;
  ai.pApplicationName="gguf2bin2";
  /* 1.0 directo y UNA sola llamada: este driver viejo crashea si se llama
     dos veces tras un fallo (probado con reintento 1.1→1.0) */
  ai.apiVersion=VK_MAKE_API_VERSION(0,1,0,0);
  { FILE *L=fopen("vk.log","a"); fprintf(L,"D: creando instancia api=%x\n",ai.apiVersion); fclose(L); }
  VkInstanceCreateInfo ci; memset(&ci,0,sizeof ci);
  ci.sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO; ci.pApplicationInfo=&ai;
  VkResult vir=p_vkCreateInstance(&ci,VK_NULL_HANDLE,&g_inst);
  if(vir!=VK_SUCCESS){
    fprintf(stderr,"vk: fallo instancia (VkResult %d)\n",(int)vir);
    { FILE *L=fopen("vk.log","a"); fprintf(L,"D3: fallo instancia VkResult %d\n",(int)vir); fclose(L); }
    return -1;
  }
  /* ICD crudo: las funciones de instancia SOLO se resuelven con la
     instancia ya creada (gpa(NULL,...) devuelve NULL para ellas) */
  GETF(vkEnumeratePhysicalDevices); GETF(vkGetPhysicalDeviceProperties);
  GETF(vkGetPhysicalDeviceMemoryProperties); GETF(vkCreateDevice);
  GETF(vkDestroyInstance);
  u32 n=0;
  if(p_vkEnumeratePhysicalDevices(g_inst,&n,VK_NULL_HANDLE)||!n){ fprintf(stderr,"vk: 0 GPUs\n"); return -1; }
  VkPhysicalDevice *pds=malloc(n*sizeof(void*));
  p_vkEnumeratePhysicalDevices(g_inst,&n,pds);
  for(u32 i=0;i<n;i++){
    VkPhysicalDeviceProperties pr; p_vkGetPhysicalDeviceProperties(pds[i],&pr);
    fprintf(stderr,"vk[%u]: %s (vendor %04x tipo %u)\n",i,pr.deviceName,pr.vendorID,pr.deviceType);
    VkPhysicalDeviceMemoryProperties mp; p_vkGetPhysicalDeviceMemoryProperties(pds[i],&mp);
    for(u32 h=0;h<mp.memoryHeapCount;h++)
      fprintf(stderr,"   heap %u: %llu MB%s\n",h,(unsigned long long)(mp.memoryHeaps[h].size>>20),
        (mp.memoryHeaps[h].flags&VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)?" [device-local]":"");
  }
  g_pd=pds[0];
  for(u32 i=0;i<n;i++){
    VkPhysicalDeviceProperties pr; p_vkGetPhysicalDeviceProperties(pds[i],&pr);
    if(pr.deviceType==VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU){ g_pd=pds[i]; break; }
  }
  { VkPhysicalDeviceProperties pr; p_vkGetPhysicalDeviceProperties(g_pd,&pr);
    strncpy(g_name,pr.deviceName,255); g_name[255]=0; }
  { VkPhysicalDeviceMemoryProperties mp; p_vkGetPhysicalDeviceMemoryProperties(g_pd,&mp);
    for(u32 h=0;h<mp.memoryHeapCount;h++)
      if(mp.memoryHeaps[h].flags&VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) g_vram=mp.memoryHeaps[h].size; }
  { u32 nf=0;
    PFN_vkGetPhysicalDeviceQueueFamilyProperties qfp=(PFN_vkGetPhysicalDeviceQueueFamilyProperties)p_vkGetInstanceProcAddr((VkInstance)g_inst,"vkGetPhysicalDeviceQueueFamilyProperties");
    qfp(g_pd,&nf,VK_NULL_HANDLE);
    VkQueueFamilyProperties *qf=malloc(nf*sizeof(*qf)); qfp(g_pd,&nf,qf);
    g_qfam=99; for(u32 f2=0;f2<nf;f2++) if(qf[f2].queueFlags&1){ g_qfam=f2; break; }
    free(qf);
    if(g_qfam==99){ fprintf(stderr,"vk: sin cola de computo\n"); free(pds); return -1; }
    float pr2=1.f;
    VkDeviceQueueCreateInfo qc={VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,VK_NULL_HANDLE,0,g_qfam,1,&pr2};
    VkDeviceCreateInfo dc={VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,VK_NULL_HANDLE,0,1,&qc,0,VK_NULL_HANDLE,0,VK_NULL_HANDLE};
    if(p_vkCreateDevice(g_pd,&dc,VK_NULL_HANDLE,&g_dev)){ fprintf(stderr,"vk: fallo dispositivo\n"); free(pds); return -1; }
    { PFN_vkGetDeviceProcAddr gdpa=(PFN_vkGetDeviceProcAddr)p_vkGetInstanceProcAddr(g_inst,"vkGetDeviceProcAddr");
      if(!gdpa){ fprintf(stderr,"vk: sin vkGetDeviceProcAddr\n"); free(pds); return -1; }
      PFN_vkGetDeviceQueue gdq=(PFN_vkGetDeviceQueue)gdpa(g_dev,"vkGetDeviceQueue");
      if(gdq) gdq(g_dev,g_qfam,0,&g_queue); }
    fprintf(stderr,"[vk] dispositivo listo: %s\n",g_name);
  }
  p_GetDeviceProcAddr=(PFN_vkGetDeviceProcAddr)p_vkGetInstanceProcAddr((VkInstance)g_inst,"vkGetDeviceProcAddr");
  free(pds);
  return 0;
}
void vk_shutdown(void){
  if(g_dev&&p_vkDestroyDevice){
    if(g_pipe_ok){
      PFN_vkDestroyPipeline p_DP=(PFN_vkDestroyPipeline)p_GetDeviceProcAddr(g_dev,"vkDestroyPipeline");
      PFN_vkDestroyDescriptorPool p_DDP=(PFN_vkDestroyDescriptorPool)p_GetDeviceProcAddr(g_dev,"vkDestroyDescriptorPool");
      PFN_vkDestroyDescriptorSetLayout p_DDSL=(PFN_vkDestroyDescriptorSetLayout)p_GetDeviceProcAddr(g_dev,"vkDestroyDescriptorSetLayout");
      PFN_vkDestroyPipelineLayout p_DPL=(PFN_vkDestroyPipelineLayout)p_GetDeviceProcAddr(g_dev,"vkDestroyPipelineLayout");
      PFN_vkDestroyCommandPool p_DCP=(PFN_vkDestroyCommandPool)p_GetDeviceProcAddr(g_dev,"vkDestroyCommandPool");
      if(g_pipe&&p_DP) p_DP(g_dev,g_pipe,VK_NULL_HANDLE);
      if(g_dpool&&p_DDP) p_DDP(g_dev,g_dpool,VK_NULL_HANDLE);
      if(g_dlayout&&p_DDSL) p_DDSL(g_dev,g_dlayout,VK_NULL_HANDLE);
      if(g_playout&&p_DPL) p_DPL(g_dev,g_playout,VK_NULL_HANDLE);
      if(g_cpool&&p_DCP) p_DCP(g_dev,g_cpool,VK_NULL_HANDLE);
      g_pipe=VK_NULL_HANDLE; g_dpool=VK_NULL_HANDLE; g_dlayout=VK_NULL_HANDLE;
      g_playout=VK_NULL_HANDLE; g_cpool=VK_NULL_HANDLE; g_cb=VK_NULL_HANDLE;
      g_pipe_ok=0;
    }
    head_free();
    p_vkDestroyDevice(g_dev,VK_NULL_HANDLE);
  }
  if(g_inst&&p_vkDestroyInstance) p_vkDestroyInstance(g_inst,VK_NULL_HANDLE);
  g_dev=VK_NULL_HANDLE; g_inst=VK_NULL_HANDLE;
}
const char *vk_device_name(void){ return g_name; }
u64 vk_vram(void){ return g_vram; }
int vk_device_ok(void){ return g_dev!=VK_NULL_HANDLE; }

/* ── fase 2: head GEMV Q4_0S ── */

static u32 find_memtype(VkMemoryRequirements *mr, VkMemoryPropertyFlags flags){
  VkPhysicalDeviceMemoryProperties mp; p_vkGetPhysicalDeviceMemoryProperties(g_pd,&mp);
  for(u32 t=0;t<mp.memoryTypeCount;t++)
    if((mr->memoryTypeBits&(1u<<t)) && (mp.memoryTypes[t].propertyFlags&flags)==flags) return t;
  return 99;
}
static void head_free(void){
  if(!g_dev) return;
  PFN_vkDestroyBuffer p_DBuf=(PFN_vkDestroyBuffer)p_GetDeviceProcAddr(g_dev,"vkDestroyBuffer");
  PFN_vkFreeMemory p_FMem=(PFN_vkFreeMemory)p_GetDeviceProcAddr(g_dev,"vkFreeMemory");
  VkBuffer bufs[3]={g_wbuf,g_xbuf,g_obuf};
  VkDeviceMemory mems[3]={g_wmem,g_xmem,g_omem};
  for(int k=0;k<3;k++){
    if(bufs[k]&&p_DBuf) p_DBuf(g_dev,bufs[k],VK_NULL_HANDLE);
    if(mems[k]&&p_FMem) p_FMem(g_dev,mems[k],VK_NULL_HANDLE);
  }
  g_wbuf=g_xbuf=g_obuf=VK_NULL_HANDLE;
  g_wmem=g_xmem=g_omem=VK_NULL_HANDLE;
  g_xmap=VK_NULL_HANDLE; g_omap=VK_NULL_HANDLE;
  g_head_ok=0;
}

int vk_head_upload(const u8 *weights, i32 n, i32 rows){
  if(n%32||n%256){ fprintf(stderr,"vkhead: dim %d no multiple de 256\n",n); return -1; }
  if(!g_dev) return -1;
  g_rows=(u32)rows; g_rows_total=(u32)rows; g_nsb=(u32)(n/256); g_n_head=n; g_row_u32s=33;
  size_t wbytes=(size_t)rows*g_nsb*33*4, xb=(size_t)n*4, ob=(size_t)rows*4;
  g_wbytes=wbytes;
  GETD(vkCreateBuffer);
  VkBufferCreateInfo wb={VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,VK_NULL_HANDLE,0,(VkDeviceSize)wbytes,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,VK_SHARING_MODE_EXCLUSIVE};
  VkBufferCreateInfo sb2=wb; sb2.size=(VkDeviceSize)xb;
  VkBufferCreateInfo ob2=wb; ob2.size=(VkDeviceSize)ob;
  if(p_vkCreateBuffer(g_dev,&wb,VK_NULL_HANDLE,&g_wbuf)) return -1;
  if(p_vkCreateBuffer(g_dev,&sb2,VK_NULL_HANDLE,&g_xbuf)) return -1;
  if(p_vkCreateBuffer(g_dev,&ob2,VK_NULL_HANDLE,&g_obuf)) return -1;
  GETD(vkAllocateMemory); GETD(vkBindBufferMemory);
  GETD(vkMapMemory); GETD(vkUnmapMemory);
  PFN_vkGetBufferMemoryRequirements p_GBR=(PFN_vkGetBufferMemoryRequirements)p_GetDeviceProcAddr(g_dev,"vkGetBufferMemoryRequirements");
  VkBuffer bufs[3]={g_wbuf,g_xbuf,g_obuf};
  VkDeviceMemory *mm[3]={&g_wmem,&g_xmem,&g_omem};
  for(int k=0;k<3;k++){
    VkMemoryRequirements mr; p_GBR(g_dev,bufs[k],&mr);
    u32 mt=find_memtype(&mr,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if(mt==99){ fprintf(stderr,"vkhead: sin memtype host-visible\n"); head_free(); return -1; }
    VkMemoryAllocateInfo mai={VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,VK_NULL_HANDLE,mr.size,mt};
    if(p_vkAllocateMemory(g_dev,&mai,VK_NULL_HANDLE,mm[k])){ head_free(); return -1; }
    if(p_vkBindBufferMemory(g_dev,bufs[k],*mm[k],0)){ head_free(); return -1; }
  }
  if(p_vkMapMemory(g_dev,g_xmem,0,(VkDeviceSize)xb,0,(void**)&g_xmap)) { head_free(); return -1; }
  if(p_vkMapMemory(g_dev,g_omem,0,(VkDeviceSize)ob,0,(void**)&g_omap)) { head_free(); return -1; }
  void *wm; if(p_vkMapMemory(g_dev,g_wmem,0,(VkDeviceSize)wbytes,0,&wm)) { head_free(); return -1; }
  i32 nb=n/32;
  for(i32 r=0;r<rows;r++){
    const u8 *src=weights+(size_t)r*(size_t)(nb/8)*130;
    uint32_t *dbase=(uint32_t*)wm+(size_t)r*(size_t)g_nsb*33;
    for(i32 sb=0;sb<nb/8;sb++){
      const u8 *ss=src+(size_t)sb*130; uint32_t s16; memcpy(&s16,ss,2);
      dbase[(size_t)sb*33]=s16;
      memcpy(dbase+(size_t)sb*33+1,ss+2,128);
    }
  }
  p_vkUnmapMemory(g_dev,g_wmem);
  g_head_ok=1; return 0;
}

/* Upload de pesos Q4_0 (bloques de 32, 18 B) al layout del shader q40_gemv:
   por bloque 17 u32 = escala f16 en bits bajos + 16 bytes de datos. */
int vk_head_upload_q40(const u8 *weights, i32 n, i32 rows){
  if(n%32){ fprintf(stderr,"vkhead: dim %d no multiple de 32\n",n); return -1; }
  if(!g_dev) return -1;
  g_rows=(u32)rows; g_rows_total=(u32)rows; g_nsb=(u32)(n/32); g_n_head=n; g_row_u32s=17;
  size_t wbytes=(size_t)rows*g_nsb*17*4, xb=(size_t)n*4, ob=(size_t)rows*4;
  g_wbytes=wbytes;
  GETD(vkCreateBuffer);
  VkBufferCreateInfo wb={VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,VK_NULL_HANDLE,0,(VkDeviceSize)wbytes,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,VK_SHARING_MODE_EXCLUSIVE};
  VkBufferCreateInfo sb2=wb; sb2.size=(VkDeviceSize)xb;
  VkBufferCreateInfo ob2=wb; ob2.size=(VkDeviceSize)ob;
  if(p_vkCreateBuffer(g_dev,&wb,VK_NULL_HANDLE,&g_wbuf)) return -1;
  if(p_vkCreateBuffer(g_dev,&sb2,VK_NULL_HANDLE,&g_xbuf)) return -1;
  if(p_vkCreateBuffer(g_dev,&ob2,VK_NULL_HANDLE,&g_obuf)) return -1;
  GETD(vkAllocateMemory); GETD(vkBindBufferMemory); GETD(vkMapMemory); GETD(vkUnmapMemory);
  PFN_vkGetBufferMemoryRequirements p_GBR=(PFN_vkGetBufferMemoryRequirements)p_GetDeviceProcAddr(g_dev,"vkGetBufferMemoryRequirements");
  VkBuffer bufs[3]={g_wbuf,g_xbuf,g_obuf};
  VkDeviceMemory *mm[3]={&g_wmem,&g_xmem,&g_omem};
  for(int k=0;k<3;k++){
    VkMemoryRequirements mr; p_GBR(g_dev,bufs[k],&mr);
    u32 mt=find_memtype(&mr,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if(mt==99){ fprintf(stderr,"vkhead: sin memtype host-visible\n"); head_free(); return -1; }
    VkMemoryAllocateInfo mai={VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,VK_NULL_HANDLE,mr.size,mt};
    if(p_vkAllocateMemory(g_dev,&mai,VK_NULL_HANDLE,mm[k])){ head_free(); return -1; }
    if(p_vkBindBufferMemory(g_dev,bufs[k],*mm[k],0)){ head_free(); return -1; }
  }
  void *wm; if(p_vkMapMemory(g_dev,g_wmem,0,(VkDeviceSize)wbytes,0,&wm)){ head_free(); return -1; }
  i32 nb=n/32;
  for(i32 r=0;r<rows;r++){
    const u8 *src=weights+(size_t)r*(size_t)nb*18;
    uint32_t *d=(uint32_t*)wm+(size_t)r*(size_t)g_nsb*17;
    for(i32 b=0;b<nb;b++){
      const u8 *p=src+(size_t)b*18; uint16_t s16; memcpy(&s16,p,2);
      d[(size_t)b*17]=(uint32_t)s16;
      memcpy(d+(size_t)b*17+1,p+2,16);
    }
  }
  p_vkUnmapMemory(g_dev,g_wmem);
  if(p_vkMapMemory(g_dev,g_xmem,0,(VkDeviceSize)xb,0,(void**)&g_xmap)){ head_free(); return -1; }
  if(p_vkMapMemory(g_dev,g_omem,0,(VkDeviceSize)ob,0,(void**)&g_omap)){ head_free(); return -1; }
  g_head_ok=1; return 0;
}

static int head_pipeline_spv(const char *spv_path);

int vk_head_pipeline(void){ return head_pipeline_spv("shaders/q4_gemv.spv"); }

static int head_pipeline_spv(const char *spv_path){
  if(!g_head_ok||!g_dev) return -1;
  if(g_pipe_ok) return 0;
  GETD(vkCreateShaderModule); GETD(vkCreateDescriptorSetLayout); GETD(vkCreatePipelineLayout);
  GETD(vkCreateComputePipelines); GETD(vkCreateDescriptorPool); GETD(vkAllocateDescriptorSets);
  GETD(vkUpdateDescriptorSets); GETD(vkCreateCommandPool); GETD(vkAllocateCommandBuffers);
  GETD(vkResetCommandPool); GETD(vkBeginCommandBuffer); GETD(vkCmdBindPipeline);
  GETD(vkCmdBindDescriptorSets); GETD(vkCmdPushConstants); GETD(vkCmdDispatch);
  GETD(vkEndCommandBuffer); GETD(vkQueueSubmit); GETD(vkQueueWaitIdle);
  FILE *f=fopen(spv_path,"rb");
  if(!f){ fprintf(stderr,"vkhead: falta %s\n",spv_path); return -1; }
  fseek(f,0,SEEK_END); long sz=ftell(f); fseek(f,0,SEEK_SET);
  if(sz<=0||sz%4){ fprintf(stderr,"vkhead: spv tamaño invalido (%ld)\n",sz); fclose(f); return -1; }
  uint32_t *code=malloc((size_t)sz);
  if(fread(code,1,(size_t)sz,f)!=(size_t)sz){ fclose(f); free(code); return -1; }
  fclose(f);
  VkShaderModuleCreateInfo sm={VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,VK_NULL_HANDLE,0,(size_t)sz,code};
  VkShaderModule mod;
  int rc=p_vkCreateShaderModule(g_dev,&sm,VK_NULL_HANDLE,&mod); free(code);
  if(rc) return -1;
  VkDescriptorSetLayoutBinding b[3];
  for(int i=0;i<3;i++){ b[i].binding=(uint32_t)i; b[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    b[i].descriptorCount=1; b[i].stageFlags=VK_SHADER_STAGE_COMPUTE_BIT; b[i].pImmutableSamplers=VK_NULL_HANDLE; }
  VkDescriptorSetLayoutCreateInfo dsl={VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,VK_NULL_HANDLE,0,3,b};
  if(p_vkCreateDescriptorSetLayout(g_dev,&dsl,VK_NULL_HANDLE,&g_dlayout)) return -1;
  VkPushConstantRange pcr={VK_SHADER_STAGE_COMPUTE_BIT,0,16};
  VkPipelineLayoutCreateInfo pli={VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,VK_NULL_HANDLE,0,1,&g_dlayout,1,&pcr};
  if(p_vkCreatePipelineLayout(g_dev,&pli,VK_NULL_HANDLE,&g_playout)) return -1;
  VkPipelineShaderStageCreateInfo st={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,VK_NULL_HANDLE,0,VK_SHADER_STAGE_COMPUTE_BIT,mod,"main",VK_NULL_HANDLE};
  VkComputePipelineCreateInfo pci={VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,VK_NULL_HANDLE,0,st,g_playout,VK_NULL_HANDLE,-1};
  if(p_vkCreateComputePipelines(g_dev,VK_NULL_HANDLE,1,&pci,VK_NULL_HANDLE,&g_pipe)){ fprintf(stderr,"vkhead: fallo pipeline\n"); return -1; }
  VkDescriptorPoolSize ps={VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,3};
  VkDescriptorPoolCreateInfo dp={VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,VK_NULL_HANDLE,0,1,1,&ps};
  if(p_vkCreateDescriptorPool(g_dev,&dp,VK_NULL_HANDLE,&g_dpool)) return -1;
  VkDescriptorSetAllocateInfo dai={VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,VK_NULL_HANDLE,g_dpool,1,&g_dlayout};
  if(p_vkAllocateDescriptorSets(g_dev,&dai,&g_dset)) return -1;
  VkDescriptorBufferInfo db[3]={{g_wbuf,0,(VkDeviceSize)g_wbytes},{g_xbuf,0,(VkDeviceSize)(size_t)g_n_head*4},{g_obuf,0,(VkDeviceSize)(size_t)g_rows_total*4}};
  VkWriteDescriptorSet w[3];
  for(int i=0;i<3;i++){ w[i].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w[i].pNext=VK_NULL_HANDLE;
    w[i].dstSet=g_dset; w[i].dstBinding=(uint32_t)i; w[i].dstArrayElement=0; w[i].descriptorCount=1;
    w[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[i].pImageInfo=VK_NULL_HANDLE; w[i].pBufferInfo=&db[i]; w[i].pTexelBufferView=VK_NULL_HANDLE; }
  p_vkUpdateDescriptorSets(g_dev,3,w,0,VK_NULL_HANDLE);
  VkCommandPoolCreateInfo cpi={VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,VK_NULL_HANDLE,VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,g_qfam};
  if(p_vkCreateCommandPool(g_dev,&cpi,VK_NULL_HANDLE,&g_cpool)) return -1;
  VkCommandBufferAllocateInfo cai={VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,VK_NULL_HANDLE,g_cpool,VK_COMMAND_BUFFER_LEVEL_PRIMARY,1};
  if(p_vkAllocateCommandBuffers(g_dev,&cai,&g_cb)) return -1;
  g_pipe_ok=1; return 0;
}

/* dispatch de un rango de filas [row0,row0+rows) del GEMV */
static int head_dispatch(u32 row0,u32 rows){
  PFN_vkResetCommandBuffer rst=(PFN_vkResetCommandBuffer)p_GetDeviceProcAddr(g_dev,"vkResetCommandBuffer");
  PFN_vkBeginCommandBuffer beg=(PFN_vkBeginCommandBuffer)p_GetDeviceProcAddr(g_dev,"vkBeginCommandBuffer");
  VkResult vr=rst(g_cb,0); if(vr) return -1;
  VkCommandBufferBeginInfo bi={VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,VK_NULL_HANDLE,VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,VK_NULL_HANDLE};
  if((vr=beg(g_cb,&bi))) return -1;
  p_vkCmdBindPipeline(g_cb,VK_PIPELINE_BIND_POINT_COMPUTE,g_pipe);
  p_vkCmdBindDescriptorSets(g_cb,VK_PIPELINE_BIND_POINT_COMPUTE,g_playout,0,1,&g_dset,0,VK_NULL_HANDLE);
  uint32_t pc[4]={rows,g_nsb,row0,g_nsb*g_row_u32s};
  p_vkCmdPushConstants(g_cb,g_playout,VK_SHADER_STAGE_COMPUTE_BIT,0,16,pc);
  p_vkCmdDispatch(g_cb,(rows+63)/64,1,1);
  p_vkEndCommandBuffer(g_cb);
  VkSubmitInfo si={VK_STRUCTURE_TYPE_SUBMIT_INFO,VK_NULL_HANDLE,0,VK_NULL_HANDLE,VK_NULL_HANDLE,1,&g_cb,0,VK_NULL_HANDLE};
  if((vr=p_vkQueueSubmit(g_queue,1,&si,VK_NULL_HANDLE))) return -1;
  return 0;
}

void vk_head_run(const f32 *x, f32 *logits){
  if(!g_pipe_ok){ memset(logits,0,(size_t)g_rows*4); return; }
  memcpy(g_xmap,x,(size_t)g_n_head*4);
  memset(g_omap,0,(size_t)g_rows*4);
  if(head_dispatch(0,g_rows)){ return; }
  p_vkQueueWaitIdle(g_queue);
  memcpy(logits,g_omap,(size_t)g_rows*4);
}
int vk_head_ready(void){ return g_pipe_ok; }

/* ══════════════════════════════════════════════════════════════════
   fase 3: DUAL BAND CPU+GPU — el driver AMD tumba el proceso con AV
   dentro de vkCreateInstance, asi que TODO el codigo Vulkan vive en
   un proceso hijo (--gpu-worker). Si el hijo crashea o cuelga, el
   padre cae a CPU-only sin enterarse el usuario.
   Protocolo binario por stdin/stdout del hijo:
     'C'                      → GEMV completo, responde vocab*4 bytes (calibracion)
     'S' u32 row0,u32 rows    → fija el rango GPU
     'R' x[dim*4]             → GEMV del rango, responde rows*4 bytes
     'Q' / EOF                → salir
   ══════════════════════════════════════════════════════════════════ */
#include "g2b.h"
#ifdef _WIN32
#include <io.h>
#include <fcntl.h>

static u32 g_wr0=0, g_wr=0;      /* rango GPU actual (hijo y padre) */

/* ── lado hijo ── */
int vk_worker_main(int argc,char **argv){
  if(argc<8) return 2;
  const char *model=argv[2];
  u64 off=strtoull(argv[3],0,10), len=strtoull(argv[4],0,10);
  i32 n=atoi(argv[5]), rows=atoi(argv[6]);
  u32 wtype=(u32)atoi(argv[7]);
  if(vk_init()) return 3;                     /* puede colgar/crashear: padre lo tolera */
  FILE *f=fopen(model,"rb");
  if(!f||fseek(f,(long)off,SEEK_SET)){ if(f)fclose(f); return 4; }
  u8 *w=malloc((size_t)len);
  if(!w||fread(w,1,(size_t)len,f)!=(size_t)len){ fclose(f); free(w); return 5; }
  fclose(f);
  int rc;
  if(wtype==T_Q4_0S){
    rc=vk_head_upload(w,n,rows);                        /* layout Q4_0S: 33 u32/sb */
    if(!rc) rc=head_pipeline_spv("shaders/q40s_gemv.spv");
  } else {
    rc=vk_head_upload_q40(w,n,rows);                    /* layout Q4_0: 17 u32/bloque */
    if(!rc) rc=head_pipeline_spv("shaders/q40_gemv.spv");
  }
  free(w);
  if(rc) return 6;
  _setmode(_fileno(stdin),_O_BINARY); _setmode(_fileno(stdout),_O_BINARY);
  fputs("VKOK\n",stdout); fflush(stdout);
  i32 done=0;
  while(!done){
    int c=fgetc(stdin);
    if(c=='C'){
      if(head_dispatch(0,g_rows)) return 8;
      p_vkQueueWaitIdle(g_queue);
      fwrite(g_omap,1,(size_t)g_rows*4,stdout); fflush(stdout);
    } else if(c=='S'){
      u32 rr[2]; if(fread(rr,1,8,stdin)!=8) break;
      g_wr0=rr[0]; g_wr=rr[1];
    } else if(c=='R'){
      if(fread(g_xmap,1,(size_t)g_n_head*4,stdin)!=(size_t)g_n_head*4) break;
      if(head_dispatch(g_wr0,g_wr)) return 8;
      p_vkQueueWaitIdle(g_queue);
      fwrite(g_omap+g_wr0,1,(size_t)g_wr*4,stdout); fflush(stdout);
    } else done=1;
  }
  return 0;
}

/* ── lado padre ── */
static HANDLE g_proc=NULL, g_inW=NULL, g_outR=NULL;
static int g_state=-1;            /* -1 sin probar, 0 apagado, 1 activo */
static int g_cal=0;
static u32 g_dtype=99;
static i32 g_vn=0, g_vocab=0;
static f32 *g_scr=NULL;
static LARGE_INTEGER g_qpf;

static int wread(void *b,u32 sz){
  u8 *p=(u8*)b; DWORD got=0;
  while(got<sz){
    DWORD r=0;
    if(!ReadFile(g_outR,p+got,sz-got,&r,NULL)||r==0) return 0;
    got+=r;
  }
  return 1;
}
static int wwrite(const void *b,u32 sz){
  const u8 *p=(const u8*)b; DWORD put=0;
  while(put<sz){
    DWORD w=0;
    if(!WriteFile(g_inW,p+put,sz-put,&w,NULL)||w==0) return 0;
    put+=w;
  }
  return 1;
}
static double now_s(void){ LARGE_INTEGER t; QueryPerformanceCounter(&t); return (double)(t.QuadPart)/(double)g_qpf.QuadPart; }
static int has_vkok(const char *b,u32 n){
  if(n<4) return 0;
  for(u32 i=0;i+4<=n;i++) if(!memcmp(b+i,"VKOK",4)) return 1;
  return 0;
}

void vk_dual_stop(void){
  if(g_proc){ TerminateProcess(g_proc,0); CloseHandle(g_proc); g_proc=NULL; }
  if(g_inW){ CloseHandle(g_inW); g_inW=NULL; }
  if(g_outR){ CloseHandle(g_outR); g_outR=NULL; }
  if(g_state==1) g_state=0;
  g_cal=0;
}
int vk_dual_active(void){ return g_state==1; }

int vk_dual_start(Model *m,const char *model_path){
  if(g_state!=-1) return g_state==1;
  QueryPerformanceFrequency(&g_qpf);
  Slot *o=slot_get(m,R_OUTPUT,-1); if(!o) o=slot_get(m,R_TOK_EMBD,-1);
  if(!o||(o->type!=T_Q4_0&&o->type!=T_Q4_0S)){ fprintf(stderr,"[gpu] head no es Q4_0/Q4_0S — dual band no disponible\n"); g_state=0; return 0; }
  g_dtype=o->type;
  char exe[MAX_PATH]; GetModuleFileNameA(NULL,exe,MAX_PATH);
  char cmd[2400];
  snprintf(cmd,sizeof cmd,"\"%s\" --gpu-worker \"%s\" %llu %llu %d %d %d",
    exe,model_path,(unsigned long long)o->off,(unsigned long long)o->nbytes,
    (int)m->c.dim,(int)m->c.vocab,(int)g_dtype);
  SECURITY_ATTRIBUTES sa={sizeof sa,NULL,TRUE};
  HANDLE inR=NULL, outR=NULL, outW=NULL;
  CreatePipe(&outR,&outW,&sa,1<<20);       /* hijo→padre: logits/handshake */
  CreatePipe(&inR,&g_inW,&sa,1<<20);       /* padre→hijo: comandos/x */
  g_outR=outR;
  SetHandleInformation(g_inW,HANDLE_FLAG_INHERIT,0);
  SetHandleInformation(g_outR,HANDLE_FLAG_INHERIT,0);
  STARTUPINFOA si; memset(&si,0,sizeof si); si.cb=sizeof si;
  si.dwFlags|=STARTF_USESTDHANDLES; si.hStdInput=inR; si.hStdOutput=outW; si.hStdError=GetStdHandle(STD_ERROR_HANDLE);
  PROCESS_INFORMATION pi; memset(&pi,0,sizeof pi);
  BOOL ok=CreateProcessA(NULL,cmd,NULL,NULL,TRUE,CREATE_NO_WINDOW,NULL,NULL,&si,&pi);
  CloseHandle(inR); CloseHandle(outW);
  if(!ok){ fprintf(stderr,"[gpu] no pude lanzar worker (%u)\n",(unsigned)GetLastError());
    CloseHandle(g_inW); CloseHandle(g_outR); g_inW=g_outR=NULL; g_state=0; return 0; }
  g_proc=pi.hProcess; CloseHandle(pi.hThread);
  /* handshake con timeout: el driver puede colgar en vez de crashear */
  char line[64]; u32 got=0; DWORD t0=GetTickCount();
  { FILE *L=fopen("vk.log","a"); fprintf(L,"P: worker lanzado, esperando VKOK\n"); fclose(L); }
  while(got<sizeof line){
    DWORD avail=0;
    if(!PeekNamedPipe(g_outR,NULL,0,NULL,&avail,NULL)){
      DWORD xc=0,le=GetLastError(); GetExitCodeProcess(pi.hProcess,&xc);
      FILE *L=fopen("vk.log","a"); fprintf(L,"P: peek fallo err=%u hijo exit=%u\n",(unsigned)le,(unsigned)xc); fclose(L);
      break;
    }
    if(avail>0){ DWORD r=0; ReadFile(g_outR,line+got,sizeof line-got,&r,NULL); got+=r;
      FILE *L=fopen("vk.log","a"); fprintf(L,"P: lei %u bytes\n",(unsigned)r); fclose(L);
      if(has_vkok(line,got)) break; }
    else{
      if(WaitForSingleObject(pi.hProcess,50)==WAIT_OBJECT_0){ FILE *L=fopen("vk.log","a"); fprintf(L,"P: hijo murio\n"); fclose(L); break; }
      if(GetTickCount()-t0>15000){ FILE *L=fopen("vk.log","a"); fprintf(L,"P: timeout 15s\n"); fclose(L); break; }
    }
  }
  if(!has_vkok(line,got)){
    fprintf(stderr,"[gpu] worker no respondio (driver roto?) — CPU only\n");
    vk_dual_stop(); return 0;
  }
  g_vn=(i32)m->c.dim; g_vocab=(i32)m->c.vocab;
  fprintf(stderr,"[gpu] worker listo\n");
  g_state=1;
  return 1;
}

int vk_head_dual(f32 *logits,const f32 *x,const u8 *w,u32 type,i32 n,i32 vocab){
  if(g_state!=1) return 0;
  if((type!=T_Q4_0&&type!=T_Q4_0S)||type!=g_dtype||n!=g_vn||vocab!=g_vocab){ vk_dual_stop(); return 0; }
  if(!g_qpf.QuadPart) QueryPerformanceFrequency(&g_qpf);
  /* calibracion una vez: tiempo CPU completo vs GPU completo → split óptimo */
  if(!g_cal){
    g_scr=(f32*)malloc((size_t)vocab*4);
    if(!g_scr){ g_state=0; return 0; }
    double t0=now_s();
    matmul_q_rows(g_scr,x,w,type,n,0,vocab);
    double tc=now_s()-t0;
    char c='C';
    double tg=0;
    if(wwrite(&c,1)&&wwrite(x,(u32)n*4)){
      t0=now_s();
      if(wread(g_scr,(u32)vocab*4)) tg=now_s()-t0;
    }
    if(tg<=0||tc<=0||tg>4.0*tc){
      fprintf(stderr,"[gpu] GPU mas lenta que CPU (tg=%.1fms tc=%.1fms) — apagada\n",tg*1e3,tc*1e3);
      vk_dual_stop(); return 0;
    }
    i32 rg=(i32)((double)vocab*tc/(tc+tg));
    if(rg<64) rg=64; if(rg>vocab-64) rg=vocab-64;
    if(rg>=vocab){ fprintf(stderr,"[gpu] split degenerado — apagada\n"); vk_dual_stop(); return 0; }
    g_wr0=(u32)(vocab-rg); g_wr=(u32)rg;
    char s='S'; u32 rr[2]={g_wr0,g_wr};
    if(!wwrite(&s,1)||!wwrite(rr,8)){ vk_dual_stop(); return 0; }
    fprintf(stderr,"[gpu] dual band: cpu=[0..%u) gpu=[%u..%d)  tc=%.1fms tg=%.1fms\n",
      (unsigned)g_wr0,(unsigned)g_wr0,vocab,tc*1e3,tg*1e3);
    g_cal=1;
  }
  /* token: enviar x (no bloquea), CPU su parte mientras la GPU trabaja, luego esperar */
  char c='R';
  if(!wwrite(&c,1)||!wwrite(x,(u32)n*4)){ vk_dual_stop(); return 0; }
  matmul_q_rows(logits,x,w,type,n,0,(i32)g_wr0);
  if(!wread(logits+g_wr0,g_wr*4)){
    vk_dual_stop();
    matmul_q_rows(logits,x,w,type,n,(i32)g_wr0,vocab);   /* completar a mano */
    return 1;
  }
  return 1;
}
#else  /* !defined(_WIN32): stubs */
int vk_worker_main(int argc,char **argv){ (void)argc;(void)argv; return -1; }
int vk_dual_start(Model *m,const char *p){ (void)m;(void)p; return 0; }
void vk_dual_stop(void){}
int vk_dual_active(void){ return 0; }
int vk_head_dual(f32 *l,const f32 *x,const u8 *w,u32 t,i32 n,i32 v){ (void)l;(void)x;(void)w;(void)t;(void)n;(void)v; return 0; }
#endif
