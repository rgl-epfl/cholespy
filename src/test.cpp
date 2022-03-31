#include "triangle_solve.h"

int main(void) {

    typedef float Float;

      uint n_rows = 9;
      uint n_entries = 18;

    uint rows_h[] = {0, 1, 2, 3, 5, 7, 9, 11, 14, 18};
    uint columns_h[] = {0, 1, 2, 0, 3, 0, 4, 1, 5, 2, 6, 3, 4, 7, 2, 3, 4, 8};
    Float values_h[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f};


    SparseTriangularSolver<Float> solver(n_rows, n_entries, rows_h, columns_h, values_h, true);

    Float b_h[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};

    std::vector<Float> x_h = solver.solve(b_h);

    for (int i=0; i<9; i++)
        std::cout << x_h[i] << " ";
    std::cout << std::endl;

    return 0;
}
