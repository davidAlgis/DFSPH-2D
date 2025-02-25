import pytest
import numpy as np
from dfsph.particle import Particle, PRESSURE, VISCOSITY, EXTERNAL, SURFACE_TENSION

@pytest.fixture
def default_particle():
    """Create a particle with default parameters"""
    return Particle(
        index=0,
        position=[0.0, 0.0],
        velocity=[0.0, 0.0],
        mass=1.0,
        h=0.1,
        type_particle="fluid"
    )

@pytest.fixture
def particle_with_forces(default_particle):
    """Particle with pre-added forces"""
    default_particle.add_force(PRESSURE, [1.0, 2.0])
    default_particle.add_force(VISCOSITY, [0.5, -1.0])
    return default_particle

def test_particle_initialization(default_particle):
    """Verify correct initialization of particle attributes"""
    assert default_particle.index == 0
    np.testing.assert_array_equal(default_particle.position, [0.0, 0.0])
    assert default_particle.mass == 1.0
    assert default_particle.h == 0.1
    assert default_particle.type_particle == "fluid"
    
    # Verify forces are initialized to zero
    for force in default_particle.forces.values():
        np.testing.assert_array_equal(force, [0.0, 0.0])

def test_force_reset(particle_with_forces):
    """Test force vector reset functionality"""
    particle_with_forces.reset_forces()
    
    for force_type in particle_with_forces.forces:
        np.testing.assert_allclose(
            particle_with_forces.forces[force_type], 
            [0.0, 0.0],
            atol=1e-9
        )

def test_valid_force_addition(default_particle):
    """Test addition of valid force types"""
    # Test with numpy array
    default_particle.add_force(PRESSURE, np.array([1.5, -2.5]))
    np.testing.assert_array_equal(
        default_particle.forces[PRESSURE], 
        [1.5, -2.5]
    )
    
    # Test with Python list
    default_particle.add_force(VISCOSITY, [0.3, 0.7])
    np.testing.assert_allclose(
        default_particle.forces[VISCOSITY], 
        [0.3, 0.7],
        atol=1e-9
    )

def test_invalid_force_addition(default_particle):
    """Verify error handling for unknown force types"""
    with pytest.raises(ValueError) as exc_info:
        default_particle.add_force("invalid_force", [1.0, 2.0])
    
    assert "Unknown force type: invalid_force" in str(exc_info.value)

def test_total_force_calculation(particle_with_forces):
    """Verify total force calculation"""
    # Add third force component
    particle_with_forces.add_force(EXTERNAL, [0.5, 1.5])
    
    total = particle_with_forces.total_force()
    expected = np.array([1.0 + 0.5 + 0.5, 2.0 - 1.0 + 1.5])
    np.testing.assert_allclose(total, expected, atol=1e-9)

def test_neighbor_assignment(default_particle):
    """Test neighbor particle assignment"""
    neighbors = [
        Particle(1, [0.1, 0.1], [0.0, 0.0], 1.0, 0.1),
        Particle(2, [0.2, 0.2], [0.0, 0.0], 1.0, 0.1)
    ]
    
    default_particle.set_neighbors(neighbors)
    assert len(default_particle.neighbors) == 2
    assert default_particle.neighbors[0].index == 1
    assert default_particle.neighbors[1].position[0] == 0.2

def test_force_accumulation(default_particle):
    """Test force vector accumulation"""
    # Multiple additions to same force type
    default_particle.add_force(SURFACE_TENSION, [0.1, 0.2])
    default_particle.add_force(SURFACE_TENSION, [0.3, 0.4])
    
    np.testing.assert_allclose(
        default_particle.forces[SURFACE_TENSION],
        [0.4, 0.6],
        atol=1e-9
    )

def test_solid_particle_creation():
    """Test solid particle type handling"""
    solid_particle = Particle(
        index=3,
        position=[0.0, 0.0],
        velocity=[0.0, 0.0],
        mass=2.0,
        h=0.2,
        type_particle="solid"
    )
    assert solid_particle.type_particle == "solid"

def test_data_type_verification(default_particle):
    """Verify numpy array data types"""
    assert isinstance(default_particle.position, np.ndarray)
    assert default_particle.position.dtype == float
    
    assert isinstance(default_particle.forces[PRESSURE], np.ndarray)
    assert default_particle.forces[PRESSURE].dtype == float

