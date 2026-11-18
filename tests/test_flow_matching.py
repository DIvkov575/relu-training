import numpy as np
from relu_viz.flow_matching import Velocity,features,sample_target
from relu_viz import mig_relief_heights


def test_target_samples_only_mig_roofs():
    points=sample_target(4096,np.random.default_rng(1))
    np.testing.assert_allclose(mig_relief_heights(points*2.5),1.2,atol=2e-5)


def test_velocity_training_reduces_error_on_known_field():
    rng=np.random.default_rng(0);net=Velocity(width=24)
    x=rng.normal(size=(256,2)).astype('f4');t=rng.random((256,1),dtype='f4')
    target=np.column_stack([x[:,1]+t[:,0],-x[:,0]]).astype('f4')
    before=np.mean((net(x,t)-target)**2)
    for step in range(1,301):net.update(x,t,target,step,.003)
    assert np.mean((net(x,t)-target)**2)<before*.03


def test_scalar_and_batch_time_inputs_agree():
    x=np.ones((12,2),dtype='f4')
    np.testing.assert_allclose(features(x,.3),features(x,np.full((12,1),.3)))


def test_midpoint_integration_follows_field_without_target_samples():
    from relu_viz.flow_matching import midpoint_step
    x=np.array([[1.,2.],[-1.,0.]],dtype='f4')
    # dx/dt = t integrates exactly with midpoint evaluation.
    result=midpoint_step(lambda x,t:np.full_like(x,t),x,.2,.1)
    np.testing.assert_allclose(result,x+.025)


def test_vector_field_uses_plot_coordinate_units_and_row_y_order():
    from relu_viz.flow_field import evaluate_field
    x=np.array([-2.5,0,2.5]);y=np.array([-1.,1.])
    field=evaluate_field(lambda points,t:points+np.array([t,0]),x,y,.5)
    assert field.shape==(2,3,2)
    np.testing.assert_allclose(field[0,:,0],x+1.25)
    np.testing.assert_allclose(field[:,0,1],y)


def test_spacetime_projection_density_and_mean_velocity():
    from relu_viz.flow_spacetime import binned_projection
    paths=np.array([[[.25,.5]]*6+[[1.25,.5]]*6],dtype='f4')
    velocities=np.array([[[2.,3.]]*6+[[4.,5.]]*6],dtype='f4')
    density,mean=binned_projection(paths,velocities,0,np.array([0.,1.,2.,3.]))
    np.testing.assert_allclose(density[:,0],[.5,.5,0])
    np.testing.assert_allclose(mean[:2,0],[2.,4.])
    assert np.isnan(mean[2,0])


def test_art_paths_end_inside_the_full_mig_footprint():
    from relu_viz.flow_art import trajectories
    paths,target=trajectories(count=1024,frames=21)
    endpoint=np.column_stack([(paths[-1,:,0]-2110)/132,(600-paths[-1,:,1])/160])
    np.testing.assert_allclose(endpoint,target,atol=3e-6)
    np.testing.assert_allclose(mig_relief_heights(endpoint),1.2,atol=3e-5)
    assert np.isfinite(paths).all()
