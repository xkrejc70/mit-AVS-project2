/**
 * @file    tree_mesh_builder.h
 *
 * @author  Jan Krejci <xkrejc70@stud.fit.vutbr.cz>
 *
 * @brief   Parallel Marching Cubes implementation using OpenMP tasks + octree early elimination
 *
 * @date    Dec 2021
 **/

#ifndef TREE_MESH_BUILDER_H
#define TREE_MESH_BUILDER_H

#include "base_mesh_builder.h"

class TreeMeshBuilder : public BaseMeshBuilder
{
public:
    TreeMeshBuilder(unsigned gridEdgeSize);

protected:
    bool checkMaxDepth(unsigned mGridSize);
    bool checkCondition(const ParametricScalarField field, const Vec3_t<float> cubeCenter, unsigned mGridSize);
    unsigned generateOctree(const ParametricScalarField& field, unsigned mGridSize, const Vec3_t<float>& position);
    unsigned marchCubes(const ParametricScalarField &field);
    float evaluateFieldAt(const Vec3_t<float> &pos, const ParametricScalarField &field);
    void emitTriangle(const Triangle_t &triangle);
    const Triangle_t* getTrianglesArray() const { return mTriangles.data(); }

    std::vector<Triangle_t> mTriangles; ///< Temporary array of triangles

    static const unsigned OCTREE = 8;
    static const unsigned CUTOFF = 1;
};

#endif // TREE_MESH_BUILDER_H
