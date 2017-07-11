import numpy as np

class MatVecMul(object):
    
    @staticmethod
    def fwd_pass(W, v):

        z = np.dot(W,v)
   
        return z
    
    @staticmethod
    def back_pass(dz, W, v):

        dW = np.outer(dz,v)
        
        dv = np.dot(W.T,dz)
        
        return dW, dv

class ReLU(object):
    
    def __init__(self):
        
        pass
    
    def ReLU_unit(z):
        
        zr = z.copy()
        
        zr[zr<0] = 0
        
        return zr
    
    def fwd_pass(self, z):
        
        self.z = z

        return ReLU.ReLU_unit(self.z)
    
    def back_pass(self, dz):

        if isinstance(dz, float):
            
            dz_relu = 0 if self.z<=0 else dz
        
        else:
            dz_relu = dz.copy()

            dz_relu[self.z<=0] = 0
        
        return dz_relu

class Add(object):
  
    @staticmethod
    def fwd_pass(A,B):    
        
        return A+B
    
    @staticmethod
    def back_pass(dC):
        
        return dC, dC

def f_softmax(z):

    m = np.max(z)

    z_exp = np.exp(z-m)

    norm = np.sum(z_exp)

    return z_exp/norm

class Tanh(object):

    def __init__(self):
        pass

    def fwd_pass(self, z):

        self.tz = np.tanh(z)

        return self.tz

    def back_pass(self, dz):

        return dz*(1-self.tz**2)

class Softmax(object):
    
    def __init__(self):
        pass
        
    def fwd_pass(self,z):
        m = np.max(z)

        z_exp = np.exp(z-m)
        
        norm = np.sum(z_exp)
        
        self.y = z_exp/norm
        
        return self.y
    
    def back_pass(self, dz):
        
        jac = -np.outer(self.y,self.y)
        
        jac.ravel()[:jac.shape[1]**2:jac.shape[1]+1] = self.y*(1-self.y)
        
        return np.dot(jac.T,dz)

class Linear(object):
    
    @staticmethod
    def fwd_pass(z, W, b):
        
        wz = MatVecMul.fwd_pass(W,z)
        
        wz_plusb = Add.fwd_pass(wz, b)
        
        return wz_plusb
    
    @staticmethod
    def back_pass(dz, z, W):
        
        d_wz, d_b = Add.back_pass(dz)
        
        d_W, d_z = MatVecMul.back_pass(d_wz, W, z)
        
        return d_z, d_W, d_b

class CrossEntropyLossSoftmax(object):
    
    @staticmethod
    def fwd_pass(Y,t):

        loss = -np.sum(t*np.log(Y))
    
        return loss

    @staticmethod
    def back_pass(Y,t):

        return -t/Y
