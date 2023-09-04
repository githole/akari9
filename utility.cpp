#define  _CRT_SECURE_NO_WARNINGS

#include "utility.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

namespace utility
{

bool writeHDRImage(const char* filename, const Image& image)
{
    return stbi_write_hdr(filename, image.width_, image.height_, 3, image.body_.data());
}


int writePNGImage(char const* filename, int w, int h, int comp, const void* data, int stride_in_bytes)
{
    return stbi_write_png(filename, w, h, comp, data, stride_in_bytes);
}

int writeJPEGImage(char const* filename, int w, int h, int comp, const void* data, int quality)
{
    return stbi_write_jpg(filename, w, h, comp, data, quality);
}

Image loadHDRImage(const char* filename)
{
    Image image;
    int component{};
    constexpr int reqComponent{ 3 };
    const float* data{ stbi_loadf(filename, &image.width_, &image.height_, &component, reqComponent) };
    if (!data)
    {
        return {};
    }
    const size_t elementCount{ (size_t)reqComponent * image.width_ * image.height_ };
    image.body_.reserve(elementCount);
    image.body_.insert(image.body_.begin(), data, data + elementCount);
    stbi_image_free((void*)data);

    std::cout << "Loaded: " << filename << std::endl;
    return image;
}

}