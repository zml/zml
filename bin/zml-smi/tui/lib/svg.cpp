#include <lunasvg.h>

#include <new>

extern "C" void* zml_smi_render_svg(const char* data, size_t length, int width,
                                    int* height, const uint8_t** pixels) noexcept {
    try {
        auto document = lunasvg::Document::loadFromData(data, length);
        if (!document || document->width() <= 0 || document->height() <= 0)
            return nullptr;

        auto bitmap = document->renderToBitmap(width, -1);
        if (bitmap.isNull())
            return nullptr;

        // Kitty expects straight RGBA, while LunaSVG renders premultiplied ARGB.
        bitmap.convertToRGBA();
        auto result = new (std::nothrow) lunasvg::Bitmap(std::move(bitmap));
        if (!result)
            return nullptr;

        *height = result->height();
        *pixels = result->data();
        return result;
    } catch (...) {
        // C++ exceptions must not cross the Zig FFI boundary.
        return nullptr;
    }
}

extern "C" void zml_smi_free_svg(void* bitmap) noexcept {
    delete static_cast<lunasvg::Bitmap*>(bitmap);
}
