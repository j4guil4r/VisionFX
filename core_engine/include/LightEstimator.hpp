#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#if __has_include(<glm/glm.hpp>)
#include <glm/glm.hpp>
#define HAS_GLM_SUPPORT 1
#endif

struct LightEstimate {
    cv::Vec3f direction{0.0f, 1.0f, 1.0f}; // Vector 3D unitario hacia la luz dominante
    float intensity{1.0f};                 // Intensidad normalizada [0.0, 1.0]
    cv::Vec3f ambientColor{0.2f, 0.2f, 0.2f};

#if HAS_GLM_SUPPORT
    glm::vec3 toGlmDirection() const {
        return glm::vec3(direction[0], direction[1], direction[2]);
    }
#endif
};

class LightEstimator {
public:
    LightEstimator();
    explicit LightEstimator(float smoothingAlpha);
    ~LightEstimator();

    // Estima dirección e intensidad de la luz dominante desde un frame cv::Mat
    LightEstimate estimate(const cv::Mat& frame);

    cv::Vec3f estimateDirection(const cv::Mat& frame);
    float estimateIntensity(const cv::Mat& frame);

    void setSmoothingFactor(float alpha);
    const LightEstimate& getLastEstimate() const;

private:
    float smoothingAlpha_;
    LightEstimate lastEstimate_;

    cv::Vec3f computeDirectionFromLuminance(const cv::Mat& gray);
    float computeNormalizedIntensity(const cv::Mat& gray);
};
