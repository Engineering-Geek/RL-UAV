from stl import mesh
import numpy as np
import path


def count_vertices_in_stl(filepath):
    """
    Loads an STL file and counts the number of unique vertices.

    :param filepath: Path to the STL file.
    :return: Number of unique vertices in the STL file.
    """
    # Load the STL file
    stl_mesh = mesh.Mesh.from_file(filepath)

    # Extract the unique vertices
    # The data is structured with 3 vertices per triangle, so we reshape and then find unique rows (vertices)
    unique_vertices = np.unique(stl_mesh.vectors.reshape(-1, stl_mesh.vectors.shape[-1]), axis=0)

    # Return the number of unique vertices
    return len(unique_vertices)


def count_triangles_in_stl(filepath):
    """
    Loads an STL file and counts the number of triangles.

    :param filepath: Path to the STL file.
    :return: Number of triangles in the STL file.
    """
    # Load the STL file
    stl_mesh = mesh.Mesh.from_file(filepath)

    # Return the number of triangles
    return len(stl_mesh.vectors)


def test():
    directory = "/home/engineering-geek/PycharmProjects/RL-models/models/assets"
    directory = path.Path(directory)
    for file in directory.files():
        print(file.name, count_vertices_in_stl(file), count_triangles_in_stl(file))


if __name__ == "__main__":
    test()
