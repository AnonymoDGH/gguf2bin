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
static PFN_vkGetPhysicalDeviceQueueFamilyProperties p_GetQFP;
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
static f32 *g_xmap,*g_omap; static int g_head_ok,g_pipe_ok;

int vk_init(void){
#ifdef _WIN32
  { FILE *L=fopen("vk.log","a"); fprintf(L,"A: entrada\n"); fclose(L); }
  if(!getenv("VK_G2B_ALLOW_AMD") && !getenv("VK_LOADER_DRIVERS_DISABLE")) _putenv("VK_LOADER_DRIVERS_DISABLE=*amd*");
  g_dll=LoadLibraryA("vulkan-1.dll");
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
  p_GetDeviceProcAddr=(PFN_vkGetDeviceProcAddr)p_vkGetInstanceProcAddr(g_inst,"vkGetDeviceProcAddr");
  { FILE *L=fopen("vk.log","a"); fprintf(L,"C: pre-instancia\n"); fclose(L); }
  if(!p_vkCreateInstance){ fprintf(stderr,"vk: sin vkCreateInstance\n"); return -1; }
  VkApplicationInfo ai; memset(&ai,0,sizeof ai);
  ai.sType=VK_STRUCTURE_TYPE_APPLICATION_INFO;
  ai.pApplicationName="gguf2bin2"; ai.apiVersion=VK_MAKE_API_VERSION(0,1,1,0);
  { FILE *L=fopen("vk.log","a"); fprintf(L,"D: creando instancia api=%x\n",ai.apiVersion); fclose(L); }
  VkInstanceCreateInfo ci; memset(&ci,0,sizeof ci);
  ci.sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO; ci.pApplicationInfo=&ai;
  if(p_vkCreateInstance(&ci,VK_NULL_HANDLE,&g_inst)!=VK_SUCCESS){ fprintf(stderr,"vk: fallo instancia\n"); return -1; }
  u32 n=0;
  if(p_vkEnumeratePhysicalDevices(g_inst,&n,VK_NULL_HANDLE)||!n){ fprintf(stderr,"vk: 0 GPUs\n"); return -1; }
  VkPhysicalDevice *pds=malloc(n*sizeof(void*));
  p_vkEnumeratePhysicalDevices(g_inst,&n,pds);
  for(u32 i=0;i<n;i++){
    VkPhysicalDeviceProperties pr; p_vkGetPhysicalDeviceProperties(pds[i],&pr);
    printf("vk[%u]: %s (vendor %04x tipo %u)\n",i,pr.deviceName,pr.vendorID,pr.deviceType);
    VkPhysicalDeviceMemoryProperties mp; p_vkGetPhysicalDeviceMemoryProperties(pds[i],&mp);
    for(u32 h=0;h<mp.memoryHeapCount;h++)
      printf("   heap %u: %llu MB%s\n",h,(unsigned long long)(mp.memoryHeaps[h].size>>20),
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
    p_vkGetDeviceQueue(g_dev,g_qfam,0,&g_queue);
    fprintf(stderr,"[vk] dispositivo listo: %s\n",g_name);
  }
  p_GetDeviceProcAddr=(PFN_vkGetDeviceProcAddr)p_vkGetInstanceProcAddr((VkInstance)g_inst,"vkGetDeviceProcAddr");
  free(pds);
  return 0;
}
void vk_shutdown(void){
  if(g_dev&&p_vkDestroyDevice) p_vkDestroyDevice(g_dev,VK_NULL_HANDLE);
  if(g_inst&&p_vkDestroyInstance) p_vkDestroyInstance(g_inst,VK_NULL_HANDLE);
  g_dev=VK_NULL_HANDLE; g_inst=VK_NULL_HANDLE;
}
const char *vk_device_name(void){ return g_name; }
u64 vk_vram(void){ return g_vram; }
int vk_device_ok(void){ return g_dev!=VK_NULL_HANDLE; }

/* ── fase 2: head GEMV Q4_0S ── */
static VkBuffer g_wbuf,g_xbuf,g_obuf;
static VkDeviceMemory g_wmem,g_xmem,g_omem;
static VkDescriptorPool g_dpool; static VkDescriptorSetLayout g_dlayout;
static VkDescriptorSet g_dset; static VkPipelineLayout g_playout; static VkPipeline g_pipe;
static VkCommandPool g_cpool; static VkCommandBuffer g_cb;
static u32 g_rows,g_nsb; static i32 g_n_head;
static f32 *g_xmap,*g_omap; static int g_head_ok,g_pipe_ok;

int vk_head_upload(const u8 *weights, i32 n, i32 rows){
  GETD(vkCreateBuffer); GETD(vkAllocateMemory); GETD(vkBindBufferMemory);
  GETD(vkMapMemory); GETD(vkUnmapMemory);
  PFN_vkGetBufferMemoryRequirements p_GBR=(PFN_vkGetBufferMemoryRequirements)p_GetDeviceProcAddr(g_dev,"vkGetBufferMemoryRequirements");
  g_rows=(u32)rows; g_nsb=(u32)(n/256); g_n_head=n;
  size_t wbytes=(size_t)rows*g_nsb*33*4, xb=(size_t)n*4, ob=(size_t)rows*4;
  VkBufferCreateInfo wb={VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,VK_NULL_HANDLE,0,(VkDeviceSize)wbytes,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,VK_SHARING_MODE_EXCLUSIVE};
  VkBufferCreateInfo sb2=wb; sb2.size=(VkDeviceSize)xb;
  VkBufferCreateInfo ob2=wb; ob2.size=(VkDeviceSize)ob;
  if(p_vkCreateBuffer(g_dev,&wb,VK_NULL_HANDLE,&g_wbuf)) return -1;
  if(p_vkCreateBuffer(g_dev,&sb2,VK_NULL_HANDLE,&g_xbuf)) return -1;
  if(p_vkCreateBuffer(g_dev,&ob2,VK_NULL_HANDLE,&g_obuf)) return -1;
  VkBuffer bufs[3]={g_wbuf,g_xbuf,g_obuf};
  VkDeviceMemory *mm[3]={&g_wmem,&g_xmem,&g_omem};
  for(int k=0;k<3;k++){
    VkMemoryRequirements mr; p_GBR(g_dev,bufs[k],&mr);
    u32 mt=99; for(u32 t=0;t<32;t++) if(mr.memoryTypeBits&(1u<<t)){ mt=t; break; }
    if(mt==99) return -1;
    VkMemoryAllocateInfo mai={VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,VK_NULL_HANDLE,mr.size,mt};
    if(p_vkAllocateMemory(g_dev,&mai,VK_NULL_HANDLE,mm[k])) return -1;
    if(p_vkBindBufferMemory(g_dev,bufs[k],*mm[k],0)) return -1;
  }
  if(p_vkMapMemory(g_dev,g_xmem,0,(VkDeviceSize)xb,0,(void**)&g_xmap)) return -1;
  if(p_vkMapMemory(g_dev,g_omem,0,(VkDeviceSize)ob,0,(void**)&g_omap)) return -1;
  void *wm; if(p_vkMapMemory(g_dev,g_wmem,0,(VkDeviceSize)wbytes,0,&wm)) return -1;
  i32 nb=n/32;
  for(i32 r=0;r<rows;r++){
    const u8 *src=weights+(size_t)r*(size_t)(nb/8)*130;
    uint32_t *dbase=(uint32_t*)wm+(size_t)r*(size_t)g_nsb*33;
    for(i32 sb=0;sb<nb/8;sb++){
      const u8 *ss=src+(size_t)sb*130; f32 h; memcpy(&h,ss,2);
      memcpy(dbase+(size_t)sb*33,&h,4);
      memcpy(dbase+(size_t)sb*33+1,ss+2,128);
    }
  }
  p_vkUnmapMemory(g_dev,g_wmem);
  g_head_ok=1; return 0;
}

int vk_head_pipeline(void){
  if(!g_head_ok||!g_dev) return -1;
  if(g_pipe_ok) return 0;
  GETD(vkCreateShaderModule); GETD(vkCreateDescriptorSetLayout); GETD(vkCreatePipelineLayout);
  GETD(vkCreateComputePipelines); GETD(vkCreateDescriptorPool); GETD(vkAllocateDescriptorSets);
  GETD(vkUpdateDescriptorSets); GETD(vkCreateCommandPool); GETD(vkAllocateCommandBuffers);
  GETD(vkResetCommandPool); GETD(vkBeginCommandBuffer); GETD(vkCmdBindPipeline);
  GETD(vkCmdBindDescriptorSets); GETD(vkCmdPushConstants); GETD(vkCmdDispatch);
  GETD(vkEndCommandBuffer); GETD(vkQueueSubmit); GETD(vkQueueWaitIdle);
  FILE *f=fopen("shaders/q4_gemv.spv","rb");
  if(!f){ fprintf(stderr,"vkhead: falta shaders/q4_gemv.spv\n"); return -1; }
  fseek(f,0,SEEK_END); long sz=ftell(f); fseek(f,0,SEEK_SET);
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
  VkPushConstantRange pcr={VK_SHADER_STAGE_COMPUTE_BIT,0,8};
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
  VkDescriptorBufferInfo db[3]={{g_wbuf,0,(VkDeviceSize)(g_rows*(size_t)g_nsb*33*4)},{g_xbuf,0,(VkDeviceSize)(size_t)g_n_head*4},{g_obuf,0,(VkDeviceSize)(size_t)g_rows*4}};
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

void vk_head_run(const f32 *x, f32 *logits){
  if(!g_pipe_ok){ memset(logits,0,(size_t)g_rows*4); return; }
  memcpy(g_xmap,x,(size_t)g_n_head*4);
  memset(g_omap,0,(size_t)g_rows*4);
  PFN_vkResetCommandBuffer rst=(PFN_vkResetCommandBuffer)p_GetDeviceProcAddr(g_dev,"vkResetCommandBuffer");
  PFN_vkBeginCommandBuffer beg=(PFN_vkBeginCommandBuffer)p_GetDeviceProcAddr(g_dev,"vkBeginCommandBuffer");
  if(rst(g_cb,0)){ return; }
  VkCommandBufferBeginInfo bi={VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,VK_NULL_HANDLE,VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,VK_NULL_HANDLE};
  if(beg(g_cb,&bi)) return;
  p_vkCmdBindPipeline(g_cb,VK_PIPELINE_BIND_POINT_COMPUTE,g_pipe);
  p_vkCmdBindDescriptorSets(g_cb,VK_PIPELINE_BIND_POINT_COMPUTE,g_playout,0,1,&g_dset,0,VK_NULL_HANDLE);
  uint32_t pc[2]={g_rows,g_nsb};
  p_vkCmdPushConstants(g_cb,g_playout,VK_SHADER_STAGE_COMPUTE_BIT,0,8,pc);
  p_vkCmdDispatch(g_cb,(g_rows+63)/64,1,1);
  p_vkEndCommandBuffer(g_cb);
  VkSubmitInfo si={VK_STRUCTURE_TYPE_SUBMIT_INFO,VK_NULL_HANDLE,0,VK_NULL_HANDLE,VK_NULL_HANDLE,1,&g_cb,0,VK_NULL_HANDLE};
  p_vkQueueSubmit(g_queue,1,&si,VK_NULL_HANDLE);
  p_vkQueueWaitIdle(g_queue);
  memcpy(logits,g_omap,(size_t)g_rows*4);
}
int vk_head_ready(void){ return g_pipe_ok; }
