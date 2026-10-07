#include "LightEstimator.hpp"
#include <algorithm>
#include <cmath>

LightEstimator::LightEstimator()
    : smoothingAlpha_(0.85f),
      lastEstimate_{cv::Vec3f(0.0f, 1.0f, 1.0f), 1.0f, cv::Vec3f(0.2f, 0.2f, 0.2f)} {}

LightEstimator::LightEstimator(float smoothingAlpha)
    : smoothingAlpha_(std::clamp(smoothingAlpha, 0.0f, 1.0f)),
      lastEstimate_{cv::Vec3f(0.0f, 1.0f, 1.0f), 1.0f, cv::Vec3f(0.2f, 0.2f, 0.2f)} {}

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

    cv::Vec3f rawDirection = computeDirectionFromLuminance(gray);
    float rawIntensity = computeNormalizedIntensity(gray);

    // Suavizado temporal exponencial
    lastEstimate_.direction = cv::normalize(
        lastEstimate_.direction * smoothingAlpha_ + rawDirection * (1.0f - smoothingAlpha_)
    );
    lastEstimate_.intensity = lastEstimate_.intensity * smoothingAlpha_ + rawIntensity * (1.0f - smoothingAlpha_);

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
    // Estimación de sesgo direccional mediante el centroide de luminosidad
    cv::Moments m = cv::moments(gray, false);
    if (m.m00 <= 1e-5) {
        return cv::Vec3f(0.0f, 1.0f, 1.0f);
    }

    float cx = static_cast<float>(m.m10 / m.m00);
    float cy = static_cast<float>(m.m01 / m.m00);

    float normX = (cx / static_cast<float>(gray.cols) - 0.5f) * 2.0f;
    float normY = -(cy / static_cast<float>(gray.rows) - 0.5f) * 2.0f;
    float normZ = 1.0f;

    return cv::normalize(cv::Vec3f(normX, normY, normZ));
}

float LightEstimator::computeNormalizedIntensity(const cv::Mat& gray) {
    cv::Scalar meanVal = cv::mean(gray);
    return std::clamp(static_cast<float>(meanVal[0] / 255.0), 0.0f, 1.0f);
}
