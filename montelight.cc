// ==Montelight==
// Tegan Brennan, Stephen Merity, Taiyo Wilson
#include <cmath>
#include <string>
#include <fstream>

#define EPSILON 0.01f

using namespace std;

struct Vector {
  double x, y, z;
  //
  Vector(const Vector &o) : x(o.x), y(o.y), z(o.z) {}
  Vector(double x_=0, double y_=0, double z_=0) : x(x_), y(y_), z(z_) {}
  inline Vector operator-(const Vector &o) const {
    return Vector(x - o.x, y - o.y, z - o.z);
  }
  inline double dot(const Vector &o) const {
    return x * o.x + y * o.y + z*o.z;
  }
};

struct Ray {
  Vector origin, direction;
  Ray(const Vector &o_, const Vector &d_) : origin(o_), direction(d_) {}
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
  virtual double intersects(const Ray &r) const { return 0; }
};

struct Sphere : Shape {
  Vector center, color;
  double radius;
  //
  Sphere(const Vector center_, double radius_, const Vector color_) :
    Shape(color_), center(center_), radius(radius_) {}
  double intersects(const Ray &r) const {
    // Find if, and at what distance, the ray intersects
    // Equation follows from solving quadratic equation of (r - c) ^ 2
    // http://wiki.cgsociety.org/index.php/Ray_Sphere_Intersection
    Vector offset = r.origin - center;
    double a = r.direction.dot(r.direction);
    double b = 2 * offset.dot(r.direction);
    double c = offset.dot(offset) - radius * radius;
    // Find discriminant for use in quadratic equation (b^2 - 4ac)
    double disc = b * b - 4 * a * c;
    // If the discriminant is negative, there are no real roots
    // (ray misses sphere)
    if (disc < 0) {
      return 0;
    }
    // The smallest positive root is the closest intersection point
    disc = sqrt(disc);
    double t = - b - disc;
    if (t > EPSILON) {
      return t;
    }
    t = - b + disc;
    if (t > EPSILON) {
      return t;
    }
    return 0;
  }
};

int main(int argc, const char *argv[]) {
  // Initialize the image
  int w = 256, h = 256;
  Image img(w, h);
  // Set up the scene
  Shape *scene[] = {
    new Sphere(Vector(100, 40, 100), 10, Vector(75, 200, 75)), // Green
    new Sphere(Vector(40, 200, 100), 20, Vector(275, 75, 100)), // Red
    new Sphere(Vector(190, 170, 300), 180, Vector(125, 75, 300)) // Blue
  };
  // Set up the camera
  // Take a set number of samples per pixel
  for (int samples = 0; samples < 1; ++samples) {
    // For each pixel, sample a ray in that direction
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        // Calculate the direction of the camera ray
        Ray ray = Ray(Vector(x, y, 0), Vector(0, 0, 1));
        // Check for intersection with objects in scene
        Vector color;
        double closest = 1e20;
        for (auto obj : scene) {
          double hit = obj->intersects(ray);
          if (hit > 0 && hit < closest) {
            color = obj->color;
            closest = hit;
          }
        }
        // Add result of sample to image
        img.setPixel(x, y, color);
      }
    }
  }
  // Save the resulting raytraced image
  img.save("render");
  return 0;
}
