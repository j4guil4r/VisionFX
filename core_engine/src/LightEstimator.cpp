#include "LightEstimator.hpp"
#include <algorithm>
#include <cmath>

namespace {
struct WeightLUT {
    float weights[256];
    WeightLUT() {
        for (int i = 0; i < 256; ++i) {
            float norm = static_cast<float>(i) / 255.0f;
            // Ponderación continua suave:
            // Permite que la luz ambiental de la habitación oriente el vector,
            // pero si entra una linterna o foco brillante, absorbe el centroide.
            weights[i] = std::pow(norm, 2.5f);
        }
    }
};

static const WeightLUT kWeightLUT;
} // namespace

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

    // Paso 1: Encontrar el valor máximo de brillo en la escena
    uint8_t maxVal = 0;
    // Muestreo rápido de zancada (stride) para encontrar el pico en microsegundos
    for (int y = 0; y < rows; y += 4) {
        const uint8_t* rowPtr = gray.ptr<uint8_t>(y);
        for (int x = 0; x < cols; x += 4) {
            if (rowPtr[x] > maxVal) maxVal = rowPtr[x];
        }
    }

    // Umbral dinámico: solo el 20% superior más brillante forma la fuente
    // Evita que el fondo del cuarto arrastre el centroide
    uint8_t dynamicThreshold = (maxVal > 80) ? static_cast<uint8_t>(maxVal * 0.80f) : 255;

    double m00 = 0.0;
    double m10 = 0.0;
    double m01 = 0.0;
    int brightPixels = 0;

    for (int y = 0; y < rows; ++y) {
        const uint8_t* rowPtr = gray.ptr<uint8_t>(y);
        for (int x = 0; x < cols; ++x) {
            uint8_t val = rowPtr[x];
            if (val >= dynamicThreshold) {
                float w = static_cast<float>(val - dynamicThreshold + 1);
                m00 += w;
                m10 += x * w;
                m01 += y * w;
                brightPixels++;
            }
        }
    }

    // 2. Dirección libre: ahora normX y normY sí alcanzan ±0.9
    cv::Vec3f rawDirection(0.0f, 0.35f, 0.85f);
    if (m00 > 1.0) {
        float cx = static_cast<float>(m10 / m00);
        float cy = static_cast<float>(m01 / m00);

        float normX = (cx / static_cast<float>(cols) - 0.5f) * 2.0f;
        float normY = -(cy / static_cast<float>(rows) - 0.5f) * 2.0f;

        // Proyección hemisférica: si X o Y crecen, Z DECAE
        float r2 = normX * normX + normY * normY;
        float normZ = (r2 < 1.0f) ? std::sqrt(1.0f - r2) : 0.05f;

        rawDirection = cv::normalize(cv::Vec3f(normX, normY, normZ));
    }

    // 3. Intensidad reactiva
    float rawIntensity = (maxVal > 240) ? 1.0f : std::clamp(static_cast<float>(maxVal) / 255.0f + 0.1f, 0.35f, 0.85f);

    // Suavizado EMA
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