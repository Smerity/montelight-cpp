// ==Montelight==
// Tegan Brennan, Stephen Merity, Taiyo Wilson
#include <cmath>
#include <string>
#include <iostream>
#include <fstream>
#include <vector>

#define EPSILON 0.001f

using namespace std;

struct Vector {
  double x, y, z;
  //
  Vector(const Vector &o) : x(o.x), y(o.y), z(o.z) {}
  Vector(double x_=0, double y_=0, double z_=0) : x(x_), y(y_), z(z_) {}
  inline Vector operator+(const Vector &o) const {
    return Vector(x + o.x, y + o.y, z + o.z);
  }
  inline Vector operator-(const Vector &o) const {
    return Vector(x - o.x, y - o.y, z - o.z);
  }
  inline Vector operator*(const Vector &o) const {
    return Vector(x * o.x, y * o.y, z * o.z);
  }
  inline Vector operator/(double o) const {
    return Vector(x / o, y / o, z / o);
  }

  inline Vector operator*(double o) const {
    return Vector(x * o, y * o, z * o);
  }
  inline double dot(const Vector &o) const {
    return x * o.x + y * o.y + z * o.z;
  }
  inline Vector &norm(){
    return *this = *this * (1 / sqrt(x * x + y * y + z * z));
  }
  inline Vector cross(Vector &o){
    return Vector(y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x);
  }
  inline double max() {
    return fmax(x, fmax(y, z));
  }
};

struct Ray {
  Vector origin, direction;
  Ray(const Vector &o_, const Vector &d_) : origin(o_), direction(d_) {}
};

struct Image {
  unsigned int width, height;
  Vector *pixels;
  unsigned int *samples;
  //
  Image(unsigned int w, unsigned int h) : width(w), height(h) {
    pixels = new Vector[width * height];
    samples = new unsigned int[width * height];
  }
  void setPixel(unsigned int x, unsigned int y, const Vector &v) {
    unsigned int index = (height - y - 1) * width + x;
    pixels[index] = pixels[index] + v;
    samples[index] += 1;
  }
  void save(std::string filePrefix) {
    std::string filename = filePrefix + ".ppm";
    std::ofstream f;
    f.open(filename.c_str(), std::ofstream::out);
    // PPM header: P3 => RGB, width, height, and max RGB value
    f << "P3 " << width << " " << height << " " << 255 << std::endl;
    // For each pixel, write the space separated RGB values
    for (int i=0; i < width * height; i++) {
      auto p = pixels[i] / samples[i];
      unsigned int r = fmin(255, p.x * 255), g = fmin(255, p.y * 255), b = fmin(255, p.z * 255);
      f << r << " " << g << " " << b << std::endl;
    }
  }
  ~Image() {
    delete[] pixels;
    delete[] samples;
  }
};

struct Shape {
  Vector color, emit;
  //
  Shape(const Vector color_, const Vector emit_) : color(color_), emit(emit_) {}
  virtual double intersects(const Ray &r) const { return 0; }
  virtual Vector randomPoint() const { return Vector(); }
  virtual Vector getNormal(const Vector &p) const { return Vector(); }
};

struct Sphere : Shape {
  Vector center;
  double radius;
  //
  Sphere(const Vector center_, double radius_, const Vector color_, const Vector emit_) :
    Shape(color_, emit_), center(center_), radius(radius_) {}
  double intersects(const Ray &r) const {
    // Find if, and at what distance, the ray intersects with this object
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
      return t / 2;
    }
    t = - b + disc;
    if (t > EPSILON) {
      return t / 2;
    }
    return 0;
  }
  Vector randomPoint() const {
    // TODO: get random point on sphere surface
    // (try not sampling points that are not visible in the scene)
    return center;
  }
  Vector getNormal(const Vector &p) const {
    // Normalize the normal by using radius instead of a sqrt call
    return (p - center) / radius;
  }
};

struct Tracer {
  std::vector<Shape *> scene;
  //
  Tracer(const std::vector<Shape *> &scene_) : scene(scene_) {}
  std::pair<Shape *, double> getIntersection(const Ray &r) const {
    Shape *hitObj = NULL;
    double closest = 1e20f;
    for (Shape *obj : scene) {
      double distToHit = obj->intersects(r);
      if (distToHit > 0 && distToHit < closest) {
        hitObj = obj;
        closest = distToHit;
      }
    }
    return std::make_pair(hitObj, closest);
  }
  Vector getRadiance(const Ray &r, int depth) {
    // Work out what (if anything) was hit
    auto result = getIntersection(r);
    Shape *hitObj = result.first;
    // Russian Roulette sampling based on reflectance of material
    double U = drand48();
    if (depth > 4 && (depth > 200 || U > hitObj->color.max())) {
      return Vector();
    }
    Vector hitPos = r.origin + r.direction * result.second;
    // Work out the contribution from directly sampling the emitters
    // TODO: Emitter Sampling
    Vector lightSampling;
    /*
    for (Shape *light : scene) {
      // Skip any objects that don't emit light
      if (light->emit.max() == 0) {
        continue;
      }
      Vector lightDirection = (light->randomPoint() - hitPos).norm();
      Ray rayToLight = Ray(hitPos, lightDirection);
      auto lightHit = getIntersection(rayToLight);
      if (light == lightHit.first) {
        lightSampling = light->emit * hitObj->color;
      }
    }
    */
    // Work out contribution from reflected light
    Vector norm = hitObj->getNormal(hitPos);
    // Orient the normal according to how the ray struck the object
    if (norm.dot(r.direction) > 0) {
      norm = norm * -1;
    }
    
    // TODO: get direction of reflectance for different materials
    
    // Diffuse reflection condition:
    // create orthogonal coordinate system defined by (x=u, y=v, z=norm)
    double angle = 2 * M_PI * drand48();
    double dist_cen = sqrt(drand48());
    Vector u;
    if (fabs(norm.x) > 0.1) {
      u = Vector(0, 1, 0);
    }
    else {
      u = Vector(1, 0, 0);
    }
    u = u.cross(norm).norm();
    Vector v = norm.cross(u);
    // direction of reflection
    Vector d = (u * cos(angle) * dist_cen + v * sin(angle) * dist_cen + norm * sqrt(1 - dist_cen * dist_cen)).norm();
    
    // recurse
    Vector reflected = getRadiance(Ray(hitPos, d), depth + 1);
    //
    return hitObj->emit + lightSampling + hitObj->color * reflected;
  }
};

int main(int argc, const char *argv[]) {
  // Initialize the image
  int w = 512, h = 512;
  Image img(w, h);
  // Set up the scene
  // Cornell box inspired: http://graphics.ucsd.edu/~henrik/images/cbox.html
  std::vector<Shape *> scene = {//Scene: position, radius, color, emission; not yet added: material
    new Sphere(Vector(1e5+1,40.8,81.6), 1e5f, Vector(.75,.25,.25), Vector()),//Left
    new Sphere(Vector(-1e5+99,40.8,81.6), 1e5f, Vector(.25,.25,.75), Vector()),//Rght
    new Sphere(Vector(50,40.8, 1e5), 1e5f, Vector(.75,.75,.75), Vector()),//Back
    new Sphere(Vector(50,40.8,-1e5+170), 1e5f, Vector(), Vector()),//Frnt
    new Sphere(Vector(50, 1e5, 81.6), 1e5f, Vector(.75,.75,.75), Vector()),//Botm
    new Sphere(Vector(50,-1e5+81.6,81.6), 1e5f, Vector(.75,.75,.75), Vector()),//Top
    new Sphere(Vector(27,16.5,47), 16.5f, Vector(1,1,1) * 0.9, Vector()),//Mirr
    new Sphere(Vector(73,16.5,78), 16.5f, Vector(1,1,1) * 0.9, Vector()),//Glas
    new Sphere(Vector(50,681.6-.27,81.6), 600, Vector(1,1,1) * 0.5, Vector(12,12,12)) //Light
    //new Sphere(Vector(50,65.1,81.6), 1.5, Vector(1,1,1), Vector(0.7,0.7,0.7)) //Light
  };
  Tracer tracer = Tracer(scene);
  // Set up the camera
  Ray camera = Ray(Vector(50, 52, 295.6), Vector(0, -0.042612, -1).norm());
  // Upright camera with field of view angle set by 0.5135
  Vector cx = Vector((w * 0.5135) / h, 0, 0);
  // Cross product gets the vector perpendicular to cx and the "gaze" direction
  Vector cy = (cx.cross(camera.direction)).norm() * 0.5135;
  // Take a set number of samples per pixel
  unsigned int SAMPLES = 100;
  for (int sample = 0; sample < SAMPLES; ++sample) {
    std::cout << "Taking sample " << sample << "\r" << std::flush;
    // For each pixel, sample a ray in that direction
    #pragma omp parallel for schedule(dynamic, 1)
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        // Calculate the direction of the camera ray
        Vector d = (cx * ((x / float(w)) - 0.5)) + (cy * ((y / float(h)) - 0.5)) + camera.direction;
        Ray ray = Ray(camera.origin + d * 140, d.norm());
        Vector color = tracer.getRadiance(ray, 0);
        // Add result of sample to image
        img.setPixel(x, y, color);
      }
    }
  }
  // Save the resulting raytraced image
  img.save("render");
  return 0;
}
