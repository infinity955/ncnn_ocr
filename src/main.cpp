/**
 * Multi-OCR ncnn inference - main entry point
 * Supports HunYuanOCR, GLM-OCR, and future models via OCRBase interface.
 */
#include "ocr_base.h"
#include "hunyuan_ocr.h"
#include "glm_ocr.h"
#include "paddleocr_vl.h"
#include "utf8_args.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <fstream>

#include <nlohmann/json.hpp>

// ============================================================
// Factory function implementation
// ============================================================

static std::string find_model_json(const std::string& model_dir) {
    for (const char* name : {"model.json", "hunyuanocr_model.json"}) {
        std::ifstream test(model_dir + "/" + name);
        if (test.is_open()) return name;
    }
    return "";
}

std::unique_ptr<OCRBase> create_ocr(const std::string& model_dir, int num_threads) {
    // Auto-detect model config file
    std::string cfg_name = find_model_json(model_dir);
    if (cfg_name.empty()) {
        fprintf(stderr, "Cannot open model.json in %s\n", model_dir.c_str());
        return nullptr;
    }
    std::ifstream ifs(model_dir + "/" + cfg_name);

    nlohmann::json config;
    ifs >> config;

    std::string model_type = config.value("model_type", "");

    // Also check setting.model_type for backward compatibility
    if (model_type.empty() && config.contains("setting")) {
        model_type = config["setting"].value("model_type", "");
    }

    if (model_type == "hunyuan_ocr") {
        printf("[Main] Detected HunYuanOCR model\n");
        return std::make_unique<hunyuan::HunYuanOCR>(model_dir, num_threads);
    } else if (model_type == "glm_ocr") {
        printf("[Main] Detected GLM-OCR model\n");
        return std::make_unique<GLMOCR>(model_dir, num_threads);
    } else if (model_type == "paddleocr_vl") {
        printf("[Main] Detected PaddleOCR-VL model\n");
        return std::make_unique<PaddleOCRVL>(model_dir, num_threads);
    } else {
        fprintf(stderr, "Unknown model_type '%s' in model.json\n", model_type.c_str());
        fprintf(stderr, "Supported types: hunyuan_ocr, glm_ocr\n");
        return nullptr;
    }
}

// ============================================================
// Main
// ============================================================

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
            printf("\n");
            printf("Supported models:\n");
            printf("  hunyuan_ocr  - HunYuanOCR (model_type: hunyuan_ocr in model.json)\n");
            printf("  glm_ocr      - GLM-OCR (model_type: glm_ocr in model.json)\n");
            return 0;
        }
    }
    if (image_path.empty()) { fprintf(stderr, "Usage: --image required\n"); return 1; }

    printf("========================================\n");
    printf("Multi-OCR ncnn Inference\n");
    printf("Model: %s  Image: %s  Prompt: %s\n", model_dir.c_str(), image_path.c_str(), prompt.c_str());
    printf("========================================\n\n");

    auto ocr = create_ocr(model_dir, 4);
    if (!ocr || !ocr->ok()) { fprintf(stderr, "Failed to load model!\n"); return 1; }

    auto ctx = ocr->prefill(prompt, image_path);
    if (!ctx) { fprintf(stderr, "Prefill failed!\n"); return 1; }

    printf("\nOCR Result:\n-----------\n");
    GenerateConfig cfg;
    cfg.max_new_tokens = 256;
    cfg.temperature = 0.0f;
    cfg.do_sample = false;

    ocr->generate(ctx, cfg, [](const std::string& token) {
        printf("%s", token.c_str()); fflush(stdout);
    });
    printf("\n\nDone.\n");
    return 0;
}