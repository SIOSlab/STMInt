import numpy as np
from astropy import units as u

def skew(vec):
	"""
	Return the skew symmetric matrix operator corresponding to the cross product

	Args:
		vec (numpy array (3)):
			A vector
	Returns:
		mat (numpy array (3x3)):
			The cross product matrix associated with vec
	"""
	return np.array([[0., -1.*vec[2], vec[1]],[vec[2],0.,-1.*vec[0]],[-1.*vec[1], vec[0], 0.]])

def findSTM(r0, v0, rf, vf, dt):
	"""
	Return the state transition matrix associated with this trajectory
	https://doi.org/10.2514/1.G006373 Reid Reynolds - Direct Solution of the Keplerian State Transition Matrix

	Args:
		r0 (numpy array (3)):
			Initial position vector
		v0 (numpy array (3)):
			Initial velocity vector
		rf (numpy array (3)):
			Final position vector
		vf (numpy array (3)):
			Final velocity vector
		dt (float):
			Time between the two states
	Returns:
		stm (numpy array (6x6))
	"""
	#km and s units
	mu = 1
	r0Mag = np.linalg.norm(r0)
	rfMag = np.linalg.norm(rf)
	h = np.cross(r0,v0)
	sr0 = skew(r0)
	sv0 = skew(v0)
	srf = skew(rf)
	svf = skew(vf)
	sh = skew(h)
	B=np.transpose(np.vstack([r0/np.sqrt(mu*r0Mag), r0Mag*v0/mu]))
	Y0 = np.block([[sr0, -1.*np.matmul((np.matmul(sr0, sv0)+sh), B), -1.*np.transpose([r0])],[sv0, np.matmul(mu/r0Mag**3*np.matmul(sr0,sr0)-np.matmul(sv0,sv0), B), np.transpose([v0])/2.]])	
	Yf = np.block([[srf, -1.*np.matmul((np.matmul(srf, svf)+sh), B), np.transpose([-1.*rf+3./2.*dt*vf])],[svf, np.matmul(mu/rfMag**3*np.matmul(srf,srf)-np.matmul(svf,svf), B), np.transpose([vf/2.-3./2.*mu/rfMag**3*dt*rf])]])
	return np.matmul(Yf, np.linalg.inv(Y0))