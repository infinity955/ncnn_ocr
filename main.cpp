/**
 * HunYuanOCR ncnn inference - main entry point
 */
#include "hunyuan_ocr.h"
#include "utf8_args.h"
#include <cstdio>
#include <cstdlib>
#include <string>

int main(int argc, char** argv) {
    auto args = get_utf8_args(argc, argv);
    std::string model_dir = ".";
    std::string image_path;
    std::string prompt = "检测并识别图片中的文字。";

    for (size_t i = 1; i < args.size(); i++) {
        const std::string& arg = args[i];
        if (arg == "--model" && i + 1 < args.size()) model_dir = args[++i];
        else if (arg == "--image" && i + 1 < args.size()) image_path = args[++i];
        else if (arg == "--prompt" && i + 1 < args.size()) prompt = args[++i];
        else if (arg == "--help" || arg == "-h") {
            printf("Usage: %s --model <dir> --image <img> [--prompt \"text\"]\n", args[0].c_str());
            return 0;
        }
    }
    if (image_path.empty()) { fprintf(stderr, "Usage: --image required\n"); return 1; }

    printf("========================================\n");
    printf("HunYuanOCR ncnn Inference\n");
    printf("Model: %s  Image: %s  Prompt: %s\n", model_dir.c_str(), image_path.c_str(), prompt.c_str());
    printf("========================================\n\n");

    hunyuan::HunYuanOCR ocr(model_dir, 4);
    if (!ocr.ok()) { fprintf(stderr, "Failed to load model!\n"); return 1; }

    auto ctx = ocr.prefill(prompt, image_path);
    if (!ctx) { fprintf(stderr, "Prefill failed!\n"); return 1; }

    printf("\nOCR Result:\n-----------\n");
    hunyuan::GenerateConfig cfg;
    cfg.max_new_tokens = 256;
    cfg.temperature = 0.0f;
    cfg.do_sample = false;

    ocr.generate(ctx, cfg, [](const std::string& token) {
        printf("%s", token.c_str()); fflush(stdout);
    });
    printf("\n\nDone.\n");
    return 0;
}
