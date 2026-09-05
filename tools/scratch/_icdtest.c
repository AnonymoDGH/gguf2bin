#include <string.h>
/* _icdtest.c ??? sonda minima: LoadLibrary(icd) + vkCreateInstance directo */
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan_core.h>
#include <stdio.h>
#include <windows.h>
int main(int argc,char**argv){
  if(argc<2){ printf("uso: %s <dll>\n",argv[0]); return 9; }
  HMODULE d=LoadLibraryA(argv[1]);
  if(!d){ printf("LOAD_FAIL\n"); return 1; }
  PFN_vkGetInstanceProcAddr gpa=(PFN_vkGetInstanceProcAddr)(void(*)(void))GetProcAddress(d,"vkGetInstanceProcAddr");
  if(!gpa){ printf("NO_GPA\n"); return 2; }
  VkApplicationInfo ai={VK_STRUCTURE_TYPE_APPLICATION_INFO,NULL,"t",0,NULL,0,VK_MAKE_API_VERSION(0,1,0,0)};
  VkInstanceCreateInfo ci; memset(&ci,0,sizeof ci);
  ci.sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO; ci.pApplicationInfo=&ai;
  VkInstance inst;
  PFN_vkCreateInstance crt=(PFN_vkCreateInstance)gpa(NULL,"vkCreateInstance");
  if(!crt){ printf("NO_CREATE\n"); return 3; }
  VkResult r=crt(&ci,NULL,&inst);
  printf("RESULT=%d\n",(int)r);
  return r==VK_SUCCESS?0:4;
}

