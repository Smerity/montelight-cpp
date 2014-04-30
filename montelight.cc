// ==Montelight==
// Tegan Brennan, Stephen Merity, Taiyo Wilson
#include <string>
#include <fstream>

using namespace std;

struct Vector {
  double x, y, z;
  //
  Vector(double x_=0, double y_=0, double z_=0) : x(x_), y(y_), z(z_) {}
};

struct Image {
  unsigned int width, height;
  Vector *pixels;
  //
  Image(unsigned int w, unsigned int h) : width(w), height(h) {
    pixels = new Vector[width * height];
  }
  void setPixel(unsigned int x, unsigned int y, Vector &v) {
    pixels[y * width + x] = v;
  }
  void save(std::string filePrefix) {
    std::string filename = filePrefix + ".ppm";
    std::ofstream f;
    f.open(filename.c_str(), std::ofstream::out);
    // PPM header: P3 => RGB, width, height, and max RGB value
    f << "P3 " << width << " " << height << " " << 255 << std::endl;
    // For each pixel, write the space separated RGB values
    for (int i=0; i < width * height; i++) {
      f << pixels[i].x << " " << pixels[i].y << " " << pixels[i].z << std::endl;
    }
  }
  ~Image() {
    delete[] pixels;
  }
};

int main(int argc, const char *argv[])
{
  Image img(256, 256);
  for (int y = 0; y < 256; ++y) {
    for (int x = 0; x < 256; ++x) {
      auto blue = Vector(75, 200, 75);
      int dx = x - 100, dy = y - 100;
      int r = 100;
      if (dx * dx + dy * dy < r) {
        img.setPixel(x, y, blue);
      }
    }
  }
  img.save("green");
  return 0;
}
