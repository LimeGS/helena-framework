// Exact, fail-closed self-intersection gate for Helena Framework triangle OBJs.
//
// The parser intentionally accepts only finite triangle geometry. Optional
// vertex colours are ignored, while OBJ texture/normal suffixes on faces are
// permitted. libigl + CGAL then performs exact-predicate intersection testing.

#include <igl/copyleft/cgal/is_self_intersecting.h>

#include <Eigen/Core>

#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int parse_index(const std::string &token, std::size_t vertex_count) {
  const auto slash = token.find('/');
  const std::string head = token.substr(0, slash);
  if (head.empty()) throw std::runtime_error("empty OBJ vertex index");
  const long raw = std::stol(head);
  if (raw == 0) throw std::runtime_error("OBJ indices are one-based");
  const long resolved = raw > 0 ? raw - 1 : static_cast<long>(vertex_count) + raw;
  if (resolved < 0 || resolved >= static_cast<long>(vertex_count)) {
    throw std::runtime_error("OBJ face index is out of range");
  }
  return static_cast<int>(resolved);
}

}  // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "usage: helena_mesh_self_intersection <triangle.obj>\n";
    return 2;
  }
  try {
    std::ifstream input(argv[1]);
    if (!input) throw std::runtime_error("cannot open OBJ");
    std::vector<Eigen::Vector3d> vertices;
    std::vector<Eigen::Vector3i> faces;
    std::string line;
    std::size_t line_number = 0;
    while (std::getline(input, line)) {
      ++line_number;
      std::istringstream stream(line);
      std::string kind;
      stream >> kind;
      if (kind.empty() || kind[0] == '#') continue;
      if (kind == "v") {
        double x, y, z;
        if (!(stream >> x >> y >> z) || !std::isfinite(x) ||
            !std::isfinite(y) || !std::isfinite(z)) {
          throw std::runtime_error("invalid vertex at line " +
                                   std::to_string(line_number));
        }
        vertices.emplace_back(x, y, z);
      } else if (kind == "f") {
        std::string a, b, c, extra;
        if (!(stream >> a >> b >> c) || (stream >> extra)) {
          throw std::runtime_error("non-triangle face at line " +
                                   std::to_string(line_number));
        }
        faces.emplace_back(parse_index(a, vertices.size()),
                           parse_index(b, vertices.size()),
                           parse_index(c, vertices.size()));
      }
    }
    if (vertices.empty() || faces.empty()) {
      throw std::runtime_error("OBJ has no triangle geometry");
    }
    Eigen::MatrixXd V(vertices.size(), 3);
    Eigen::MatrixXi F(faces.size(), 3);
    for (Eigen::Index i = 0; i < V.rows(); ++i) V.row(i) = vertices[i];
    for (Eigen::Index i = 0; i < F.rows(); ++i) F.row(i) = faces[i];
    const bool intersects = igl::copyleft::cgal::is_self_intersecting(V, F);
    std::cout << "{\"schema\":\"campaignx.mesh_self_intersection.v1\","
              << "\"vertices\":" << V.rows() << ",\"triangles\":" << F.rows()
              << ",\"self_intersections_present\":"
              << (intersects ? "true" : "false") << "}\n";
    return intersects ? 1 : 0;
  } catch (const std::exception &error) {
    std::cerr << "helena_mesh_self_intersection: " << error.what() << "\n";
    return 2;
  }
}
