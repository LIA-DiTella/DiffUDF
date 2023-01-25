import numpy as np

class Interpolation:
    def __init__( self, points ):
        if len(points) < 3:
            raise ValueError("Cannot interpolate 3 or less points")
        
        self.points = [points[0]] + points + [points[-1]]
        self.parametrization = lambda x : x

        try:
            self.dimension = len(points[0])
        except:
            self.dimension = 1
        
    def __getitem__(self, t):
        return self.evaluate(t)
    
    def evaluate( self, t ):
        '''
            t between [0,1] or in parametrization range. 
        '''

        t = self.parametrization(t)

        if t < 0 or t > 1:
            raise ValueError("p(t) = " + str(t) +" is out of range [0,1].")

        amountOfCurves = (len(self.points) - 3)
        
        if np.isclose(t, 1):
          return self.evaluateCurve( amountOfCurves, 1 )

        pointIndex = np.floor( t * amountOfCurves ).astype(np.uint32) + 1
        
        return self.evaluateCurve(pointIndex, ( t - (pointIndex - 1) / amountOfCurves ) * amountOfCurves )

    def evaluateCurve( self, indice, t ):

        def spline_4p( t, p_1, p0, p1, p2 ):

            return (
                t*((2-t)*t - 1)   * p_1
                + (t*t*(3*t - 5) + 2) * p0
                + t*((4 - 3*t)*t + 1) * p1
                + (t-1)*t*t         * p2 ) / 2

        return spline_4p(t, self.points[indice - 1], self.points[indice], self.points[indice + 1], self.points[indice + 2])

    def evaluateFrom( self , ts ):
        '''
            ts array-like.
        '''
        return ( self.evaluate(t) for t in ts )

    def arcLength( self, *, eps=0.01, tStart=0, tEnd=1 ):
        if tEnd - tStart <= eps:
            return np.linalg.norm( self.evaluate(tEnd) - self.evaluate(tStart))

        longitude = 0
        lastValue = self.evaluate(tStart)
        for step in np.arange(eps, tEnd + eps, eps):
            newValue = self.evaluate( step )
            longitude += np.linalg.norm( newValue - lastValue )
            lastValue = newValue

        return longitude

    def reparametrize( self, funcion ):
        self.parametrization = funcion
        return self

    def gradient( self, *, eps=0.01, tStart=0, tEnd=1, normalized=False ):
        samples = np.array( [ sample for sample in self.evaluateFrom( np.arange(tStart, tEnd, eps ))])
        ds = np.array([ np.gradient( samples.T[i], eps ) for i in range(self.dimension) ])

        if self.dimension == 3:
            if normalized:
                return Interpolation( ds.T ), Interpolation( list( map( lambda x : x / np.linalg.norm(x),  ds.T )) )
            else:
                return Interpolation( ds.T )
        else:
            return Interpolation( ds.T )

    def curvature( self ):
        '''
            k = norm( dT / ds ) = norm( dT / dt ) / norm( dC / dt )
        '''

        dC_dt, Tp = self.gradient(normalized=True)
        dTp_dt = Tp.gradient()

        return Interpolation( [ np.linalg.norm(dTp_dt[t]) / np.linalg.norm(dC_dt[t]) for t in np.linspace(0, 1, 100)] )

    def direction( self, t ):
        
        #ds = []
        #diffs = []
        #for ep in np.linspace( min(0.1, 1 - t), 0.0001, 100):
        #    ds.append( normalize(self[t + ep] - self[t] ) )
        #    if len(diffs) > 0:
        #        diffs.append( np.linalg.norm(ds[-1] - ds[-2]) )
        #return d / np.linalg.norm( d )

        if np.isclose(t , 1 ):
            return normalize( self[t] - self[t - 0.001] )

        return normalize(self[t + 0.001] - self[t])

    def basisAlong( self, points, amountOfSamples = 10 ):
        '''
            Calculates a basis along the curve, by projecting the normal vector throughout
            sample points on the curve.
            points vector should be sorted.
            By adding more samples, you get a smoother interpolation.
        '''
        
        basis = []

        dir0 = self.direction(0)
        normal1 = normalize( np.linalg.solve(np.array([
            dir0,
            np.random.uniform(0,1,3),
            np.random.uniform(0,1,3)
        ]), np.concatenate( [[0], np.random.uniform(0,1,2)]) ) ) # we calculate the first normal randomly

        firstBasis = np.array([ dir0, normal1, cross( dir0, normal1 ) ])
        basis.append(firstBasis)

        samples = np.concatenate( [ np.random.uniform( 0, 1, amountOfSamples ), points] )
        indexes = np.argsort( samples )
        
        result = []
        for index in indexes:
            diri = self.direction( samples[index] )
            normal1i = projectionToPlane( basis[-1][1], diri )
            basis.append( np.array( [ diri , normal1i, cross(diri, normal1i )  ] ))

            if index >= amountOfSamples:
                result.append( basis[-1]) # i return only those that correspond to points given

        return result

    
def normalize( a ):
    if np.allclose(a, np.zeros_like(a)):
        raise ValueError( 'Cannot normalize zeros' )

    return a / np.linalg.norm(a)

def projectionToPlane( v, n ):
    '''
        Returns the projection of v onto plane of normal n. Normalized.
    '''
    if not np.isclose( np.dot(n, n), 1) :
        raise ValueError('Normal should be normalized')

    return normalize( v - (np.dot(v,n) / np.dot(n, n)) * n )

# I use this function because there is some issue with numpy and pylance in vscode
# that marks code after using np.cross as unreachable, which is very annoying.
# By providing the explicit type of a and b you work around the problem.
def cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return normalize(np.cross(a,b))

