/**
 * @file    tree_mesh_builder.cpp
 *
 * @author  Jan Krejci <xkrejc70@stud.fit.vutbr.cz>
 *
 * @brief   Parallel Marching Cubes implementation using OpenMP tasks + octree early elimination
 *
 * @date    Dec 2021
 **/

#include <iostream>
#include <math.h>
#include <limits>

#include "tree_mesh_builder.h"

TreeMeshBuilder::TreeMeshBuilder(unsigned gridEdgeSize)
    : BaseMeshBuilder(gridEdgeSize, "Octree")
{

}

bool TreeMeshBuilder::checkMaxDepth(unsigned mGridSize) {
    return (mGridSize > CUTOFF) ? false : true;
}

bool TreeMeshBuilder::checkCondition(const ParametricScalarField field, const Vec3_t<float> cubeCenter, unsigned mGridSize) {
    float evalField = evaluateFieldAt(cubeCenter, field);
    float condition = mIsoLevel + ((sqrtf(3.0f) / 2.0f) * mGridSize * mGridResolution);

    return (evalField > condition) ? false : true;
}

unsigned TreeMeshBuilder::generateOctree(const ParametricScalarField &field, unsigned mGridSize, const Vec3_t<float> &position) {
    size_t totalCubesCount = mGridSize * mGridSize * mGridSize;
    float mGridSizeMid = mGridSize / 2.0f;
    unsigned totalTriangles = 0;
    Vec3_t<float> cubeCenter(
        (position.x + mGridSizeMid) * mGridResolution,
        (position.y + mGridSizeMid) * mGridResolution,
        (position.z + mGridSizeMid) * mGridResolution
    );

    // Check condition to continue in octree
    if (checkCondition(field, cubeCenter, mGridSize)) {
        totalTriangles = 0;

        if (checkMaxDepth(mGridSize)) {
            return buildCube(position, field);
        }

        /*
        #pragma omp parallel for \
        default(none) \
        shared(position, mGridSizeMid, field) \
        schedule(static, 32) \
        reduction(+:totalTriangles)
        */

        // Generate 8 cubes
        for (size_t i = 0; i < OCTREE; i++) {

            Vec3_t<float> nextPosition(
                position.x + (sc_vertexNormPos[i].x * mGridSizeMid),
                position.y + (sc_vertexNormPos[i].y * mGridSizeMid),
                position.z + (sc_vertexNormPos[i].z * mGridSizeMid)
            );

#pragma omp task shared(field, totalTriangles, mGridSizeMid)
#pragma omp atomic update
            totalTriangles += generateOctree(field, mGridSizeMid, nextPosition);
        }

#pragma omp taskwait
        return totalTriangles;

    } else {
        return 0.0f;
    }
}

unsigned TreeMeshBuilder::marchCubes(const ParametricScalarField &field)
{
    // Suggested approach to tackle this problem is to add new method to
    // this class. This method will call itself to process the children.
    // It is also strongly suggested to first implement Octree as sequential
    // code and only when that works add OpenMP tasks to achieve parallelism.
    
    // 1. Compute total number of cubes in the grid.
    size_t totalCubesCount = mGridSize * mGridSize * mGridSize;

    unsigned totalTriangles = 0;
    float start = 0.0f;
    Vec3_t<float> initPosition(start, start, start);

#pragma omp parallel shared(field, totalTriangles)
#pragma omp single nowait
    totalTriangles = generateOctree(field, mGridSize, initPosition);

    // 5. Return total number of triangles generated.
    return totalTriangles;
}

float TreeMeshBuilder::evaluateFieldAt(const Vec3_t<float> &pos, const ParametricScalarField &field)
{
    // NOTE: This method is called from "buildCube(...)"!

    // 1. Store pointer to and number of 3D points in the field
    //    (to avoid "data()" and "size()" call in the loop).
    const Vec3_t<float> *pPoints = field.getPoints().data();
    const unsigned count = unsigned(field.getPoints().size());

    float value = std::numeric_limits<float>::max();

    // 2. Find minimum square distance from points "pos" to any point in the
    //    field.
    for(unsigned i = 0; i < count; ++i)
    {
        float distanceSquared  = (pos.x - pPoints[i].x) * (pos.x - pPoints[i].x);
        distanceSquared       += (pos.y - pPoints[i].y) * (pos.y - pPoints[i].y);
        distanceSquared       += (pos.z - pPoints[i].z) * (pos.z - pPoints[i].z);

        // Comparing squares instead of real distance to avoid unnecessary
        // "sqrt"s in the loop.
        value = std::min(value, distanceSquared);
    }

    // 3. Finally take square root of the minimal square distance to get the real distance
    return sqrt(value);
}

void TreeMeshBuilder::emitTriangle(const BaseMeshBuilder::Triangle_t &triangle)
{
    // NOTE: This method is called from "buildCube(...)"!
  
    // Store generated triangle into vector (array) of generated triangles.
    // The pointer to data in this array is return by "getTrianglesArray(...)" call
    // after "marchCubes(...)" call ends.
#pragma omp critical(critical)
    mTriangles.push_back(triangle);
}
