#include "laplacian.h"
#include <iostream>

template <typename Float>
Laplacian<Float>::Laplacian(uint n_verts, uint n_faces, uint *faces, Float lambda) {

    // We start by building the (lower half of the) matrix in the COO format.

    // indices of nonzero entries
    std::vector<uint> ii;
    std::vector<uint> jj;

    // Heuristic based on average connectivity on a triangle mesh.
    ii.reserve(7 * n_verts);
    jj.reserve(7 * n_verts);
    std::vector<uint> col_entries;
    col_entries.resize(n_verts, 0);

    // Add one entry per edge
    for (uint i=0; i<n_faces; i++) {
        for (uint j=0; j<3; j++) {
            for (uint k=j+1; k<3; k++) {
                uint s = faces[3*i + j];
                uint d = faces[3*i + k];
                if (s > d) {
                    // L[s,d]
                    ii.push_back(s);
                    jj.push_back(d);
                    col_entries[d]++;
                } else {
                    // L[d,s]
                    ii.push_back(d);
                    jj.push_back(s);
                    col_entries[s]++;
                }
            }
        }
    }

    // Add diagonal indices
    for(uint i=0; i<n_verts; i++) {
        ii.push_back(i);
        jj.push_back(i);
        col_entries[i]++;
    }

    ii.shrink_to_fit();
    jj.shrink_to_fit();

    uint nnz = ii.size();

    // Then we convert the COO representation to CSC

    std::vector<uint> tmp_col_ptr;
    tmp_col_ptr.resize(n_faces+1, 0);
    std::vector<uint> tmp_rows;
    tmp_rows.resize(nnz, 0);

    /*
    Build the column pointer array, where tmp_col_ptr[i] is the start of the
    i-th column in the other arrays
    */
    uint cumsum=0;
    for (uint i=0; i<n_verts; i++) {
        tmp_col_ptr[i] = cumsum;
        cumsum += col_entries[i];
    }
    tmp_col_ptr[n_verts] = cumsum;

    /*
    Now move the row indices of each entry so that entries in column j are in
    positions tmp_col_ptr[j] to tmp_col_ptr[j+1]-1
    */
    for (uint i=0; i<nnz; i++) {
        uint col = jj[i];
        uint dst = tmp_col_ptr[col];

        tmp_rows[dst] = ii[i];
        tmp_col_ptr[col]++;
    }

    // Undo the modifications to tmp_col_ptr from the previous step
    for(uint i = 0, last = 0; i <= n_verts; i++){
        uint temp = tmp_col_ptr[i];
        tmp_col_ptr[i] = last;
        last = temp;
    }

    // Sort indices in each column to ease the removal of duplicates
    for (uint i=0; i<n_verts; i++) {
        std::sort(tmp_rows.begin() + tmp_col_ptr[i], tmp_rows.begin() + tmp_col_ptr[i+1]);
    }

    m_rows.reserve(nnz);
    m_data.reserve(nnz);
    m_col_ptr.resize(n_verts+1, 0);
    cumsum = 0;

    std::vector<uint> adjacency;
    adjacency.resize(n_verts, 0);
    // Remove duplicates
    for (uint col=0; col<n_verts; col++) {
        uint i = tmp_col_ptr[col];
        uint n_elements = 0;
        uint row = tmp_rows[i];
        while (i<tmp_col_ptr[col+1]) {
            if (row != col) {
                // Count unique off diag entries per row
                // We increment both indices because we only store half of the entries
                adjacency[row]++;
                adjacency[col]++;
            }
            m_rows.push_back(row);
            m_data.push_back(-lambda);
            n_elements++;
            uint previous_row = row;
            while (row == previous_row && i<tmp_col_ptr[col+1]) {
                // Ignore duplicate entries (all off-diagonal entries of the laplacian are ones)
                i++;
                row = tmp_rows[i];
            }
        }
        // Correct element count
        col_entries[col] = n_elements;
        // Correct column start pointer
        m_col_ptr[col] = cumsum;
        cumsum += n_elements;
    }
    m_col_ptr[n_verts] = cumsum;
    m_data.shrink_to_fit();
    m_rows.shrink_to_fit();

    // Set diagonal indices proper values
    for (uint j=0; j<n_verts; j++) {
        for (uint i=m_col_ptr[j]; i<m_col_ptr[j+1]; i++) {
            if (j == m_rows[i]) // diagonal element
                m_data[i] = (Float) adjacency[j] * lambda + 1.0f;
        }
    }
}

template class Laplacian<float>;
template class Laplacian<double>;
