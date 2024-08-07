#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <stdexcept>

// Function to load indices from a text file
std::vector<size_t> loadIndicesFromTxt(const std::string& file_path) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file");
    }

    std::string line;
    std::getline(file, line);
    file.close();

    // Remove square brackets from the string
    line.erase(std::remove(line.begin(), line.end(), '['), line.end());
    line.erase(std::remove(line.begin(), line.end(), ']'), line.end());

    std::stringstream ss(line);
    std::string number;
    std::vector<size_t> indices;

    // Parse numbers from the string
    while (std::getline(ss, number, ',')) {
        indices.push_back(static_cast<size_t>(std::stoul(number)));
    }

    return indices;
}
