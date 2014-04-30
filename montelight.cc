// ==Montelight==
// Tegan Brennan, Stephen Merity, Taiyo Wilson
#include <string>
#include <fstream>

using namespace std;

struct Vector {
  double x, y, z;
  //
  Vector(double x_=0, double y_=0, double z_=0) : x(x_), y(y_), z(z_) {}
  inline Vector operator-(const Vector &o) const {
    return Vector(x - o.x, y - o.y, z - o.z);
  }
  inline double dot(const Vector &o) const {
    return x * o.x + y * o.y + z*o.z;
  }
};

struct Image {
  unsigned int width, height;
  Vector *pixels;
  //
  Image(unsigned int w, unsigned int h) : width(w), height(h) {
    pixels = new Vector[width * height];
  }
  void setPixel(unsigned int x, unsigned int y, const Vector &v) {
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

struct Shape {
  Vector color;
  //
  Shape(const Vector color_) : color(color_) {}
  virtual double intersects(const Vector &point) const { return 0; }
};

struct Circle : Shape {
  Vector center, color;
  double radius;
  //
  Circle(const Vector center_, double radius_, const Vector color_) :
    Shape(color_), center(center_), radius(radius_) {}
  double intersects(const Vector &point) const {
    Vector d = point - center;
    if (d.dot(d) < radius) {
      return 1;
    }
    return 0;
  }
};

int main(int argc, const char *argv[])
{
  // Initialize the image
  int w = 256, h = 256;
  Image img(w, h);
  // Set up the scene
  Shape *scene[] = {
    new Circle(Vector(100, 40, 0), 100, Vector(75, 200, 75)), // Green
    new Circle(Vector(40, 200, 0), 400, Vector(275, 75, 100)), // Red
    new Circle(Vector(190, 170, 0), 300, Vector(125, 75, 300)) // Blue
  };
  // Set up the camera
  // Take a set number of samples per pixel
  for (int samples = 0; samples < 1; ++samples) {
    // For each pixel, sample a ray in that direction
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        // Calculate the direction of the camera ray
        // Check for intersection
        Vector ray = Vector(x, y, 0);
        for (auto obj : scene) {
          if (obj->intersects(ray) > 0) {
            //img.setPixel(x, y, Vector(75, 200, 75));
            img.setPixel(x, y, obj->color);
          }
        }
        // Add result of sample to image
      }
    }
  }
  // Save the resulting raytraced image
  img.save("render");
  return 0;
}
