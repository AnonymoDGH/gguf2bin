/* vk_stub.c — implementaciones no-op para builds sin backend Vulkan
   (Android y cualquier plataforma sin l7_vulkan.c) */
#include "g2b.h"

int  vk_worker_main(int argc, char **argv){ (void)argc; (void)argv; return -1; }
int  vk_init(void){ return -1; }
int  vk_device_ok(void){ return 0; }
void vkinfo_cmd(void){}
int  vk_head_upload(const u8 *w, i32 n, i32 rows){ (void)w;(void)n;(void)rows; return -1; }
int  vk_head_pipeline(void){ return -1; }
void vk_head_run(const f32 *x, f32 *logits){ (void)x; (void)logits; }
int  vk_head_ready(void){ return 0; }
int  vk_dual_start(Model *m, const char *path){ (void)m;(void)path; return 0; }
void vk_dual_stop(void){}
int  vk_dual_active(void){ return 0; }
int  vk_head_dual(f32 *logits, const f32 *x, const u8 *w, u32 type, i32 n, i32 vocab){
  (void)logits;(void)x;(void)w;(void)type;(void)n;(void)vocab; return 0;
}
