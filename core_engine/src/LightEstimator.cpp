#include "LightEstimator.hpp"
#include <algorithm>
#include <cmath>

/*namespace {
struct WeightLUT {
    float weights[256];
    WeightLUT() {
        for (int i = 0; i < 256; ++i) {
            float norm = static_cast<float>(i) / 255.0f;
            weights[i] = std::pow(norm, 2.5f);
        }
    }
};

static const WeightLUT kWeightLUT;
} // namespace*/

LightEstimator::LightEstimator()
    : smoothingAlpha_(0.75f),
      lastEstimate_{cv::Vec3f(0.0f, 0.35f, 0.93f), 0.55f, cv::Vec3f(0.15f, 0.15f, 0.15f)} {}

LightEstimator::LightEstimator(float smoothingAlpha)
    : smoothingAlpha_(std::clamp(smoothingAlpha, 0.0f, 1.0f)),
      lastEstimate_{cv::Vec3f(0.0f, 0.35f, 0.93f), 0.55f, cv::Vec3f(0.15f, 0.15f, 0.15f)} {}

LightEstimator::~LightEstimator() = default;

LightEstimate LightEstimator::estimate(const cv::Mat& frame) {
    if (frame.empty()) {
        return lastEstimate_;
    }

    cv::Mat gray;
    if (frame.channels() == 3) {
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    } else if (frame.channels() == 4) {
        cv::cvtColor(frame, gray, cv::COLOR_BGRA2GRAY);
    } else {
        gray = frame;
    }

    const int rows = gray.rows;
    const int cols = gray.cols;
    
    uint8_t maxVal = 0;
    for (int y = 0; y < rows; y += 4) {
        const uint8_t* rowPtr = gray.ptr<uint8_t>(y);
        for (int x = 0; x < cols; x += 4) {
            if (rowPtr[x] > maxVal) maxVal = rowPtr[x];
        }
    }

    // Umbral dinámico
    uint8_t dynamicThreshold = (maxVal > 80) ? static_cast<uint8_t>(maxVal * 0.90f) : 255;

    double m00 = 0.0;
    double m10 = 0.0;
    double m01 = 0.0;
    int brightPixels = 0;

    for (int y = 0; y < rows; ++y) {
        const uint8_t* rowPtr = gray.ptr<uint8_t>(y);
        for (int x = 0; x < cols; ++x) {
            uint8_t val = rowPtr[x];
            if (val >= dynamicThreshold) {
                float diff = static_cast<float>(val - dynamicThreshold + 1);
                float w = diff * diff; // Peso cuadrático equilibrado
                m00 += w;
                m10 += x * w;
                m01 += y * w;
                brightPixels++;
            }
        }
    }

    // Detectar si hay una fuente puntual (linterna/foco) o luz ambiental de salón
    float spotRatio = static_cast<float>(brightPixels) / static_cast<float>(rows * cols);
    bool isFlashlight = (maxVal >= 242 && spotRatio > 0.0005f && spotRatio < 0.065f);

    cv::Vec3f rawDirection(0.0f, 0.35f, 0.90f);
    if (m00 > 1.0) {
        float cx = static_cast<float>(m10 / m00);
        float cy = static_cast<float>(m01 / m00);

        // En salón normal mueve suave (1.1f); con linterna abre el ángulo completo (2.2f)
        float angleScale = isFlashlight ? 2.2f : 1.1f;
        float maxLat = isFlashlight ? 0.95f : 0.60f;

        float normX = std::clamp((cx / static_cast<float>(cols) - 0.5f) * angleScale, -maxLat, maxLat);
        float normY = std::clamp(-(cy / static_cast<float>(rows) - 0.5f) * angleScale, -maxLat, maxLat);

        float r2 = normX * normX + normY * normY;
        float minZ = isFlashlight ? 0.20f : 0.65f;
        float normZ = (r2 < 1.0f) ? std::max(std::sqrt(1.0f - r2), minZ) : minZ;

        rawDirection = cv::normalize(cv::Vec3f(normX, normY, normZ));
    }

    // Intensidad estable (0.82f) en salón normal y boost (1.25f) cuando detecta linterna
    float rawIntensity = isFlashlight
        ? 1.25f
        : std::clamp(static_cast<float>(maxVal) / 255.0f * 0.45f + 0.50f, 0.70f, 0.90f);

    // Suavizado
    float alphaDir = 0.70f;
    lastEstimate_.direction = cv::normalize(
        lastEstimate_.direction * alphaDir + rawDirection * (1.0f - alphaDir)
    );

    float alphaInt = (rawIntensity > lastEstimate_.intensity) ? 0.75f : 0.93f;
    lastEstimate_.intensity = lastEstimate_.intensity * alphaInt + rawIntensity * (1.0f - alphaInt);

    return lastEstimate_;
}

cv::Vec3f LightEstimator::estimateDirection(const cv::Mat& frame) {
    return estimate(frame).direction;
}

float LightEstimator::estimateIntensity(const cv::Mat& frame) {
    return estimate(frame).intensity;
}

void LightEstimator::setSmoothingFactor(float alpha) {
    smoothingAlpha_ = std::clamp(alpha, 0.0f, 1.0f);
}

const LightEstimate& LightEstimator::getLastEstimate() const {
    return lastEstimate_;
}

cv::Vec3f LightEstimator::computeDirectionFromLuminance(const cv::Mat& gray) {
    return estimate(gray).direction;
}

float LightEstimator::computeNormalizedIntensity(const cv::Mat& gray) {
    return estimate(gray).intensity;
}