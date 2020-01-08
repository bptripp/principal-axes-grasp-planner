import numpy as np
from scipy.signal import medfilt
from scipy.spatial import ConvexHull, Delaunay

from . import helper as h


class DepthImage:
    def __init__(self, dimg, in_meters=False):
        """
        Class to hold onto depth images and convert them to point clouds.
        The in_meters parameter is used to denote whether the input dimg is
        in meters or not. If False, we will convert the dimg to meters.
        """
        if not in_meters:
            dimg = h.depth2meters(dimg)
        self.dimg = dimg

    def depth_threshold(self, low, high=None, filter=False):
        """
        Returns a new depth image with points set to zero if they're not in the
        range given.

        If only one argument is given it will set to zero all points higher
        than the argument.

        If two arguments are given it will treat the first as the low end of the
        range and the second as the high end of the range.

        If filter is set to true we apply a median filter before returning.
        """
        if high==None:
            masked = self.dimg*(self.dimg<low)
        else:
            mask_low = self.dimg>low
            mask_high = self.dimg<high
            masked = self.dimg*mask_high*mask_low

        if filter:
            masked = medfilt(masked)
        return DepthImage(masked, in_meters=True)

    def project_points(self, x_center=312.307, y_center=245.558,
                             focal_x=473.852, focal_y=473.852):
        """
        Projects the depth map to a 3D point cloud. The optional parameters
        correspond to the intrinsic camera parameters needed to do this
        transformation. The default values provided correspond to the RealSense
        SR300.
        """
        rows, cols = self.dimg.shape
        c, r = np.meshgrid(np.arange(cols), np.arange(rows))
        z = self.dimg
        x = z*(c-x_center)/focal_x
        y = z*(r-y_center)/focal_y

        pts = np.dstack([x,y,z]).reshape([np.prod(self.dimg.shape),3])
        points = pts[pts[:,2]!=0]
        return PointCloud(points)


class PointCloud:
    def __init__(self, points, form_hull=False):
        """
        Points is an (N,3) ndarray corresponding to a point cloud. We assume
        that the units of the point cloud are meters.

        If form_hull is true we will form the convex hull of the point cloud.
        """
        self.points = points
        self.form_hull = form_hull
        if form_hull:
            self.make_convex_hull()
        return

    def make_convex_hull(self):
        self.hull = ConvexHull(self.points)
        self.make_facet_normals()
        self.delaunay = Delaunay(self.points[self.hull.vertices])
        return self.hull

    def make_facet_normals(self):
        """
        Taken from example 13 at
        https://programtalk.com/python-examples/scipy.spatial.ConvexHull/
        """
        # Assume 3d for now
        # Calculate normals from the vector cross product of the vectors defined
        # by joining points in the simplices
        vab = self.points[self.hull.simplices[:, 0]]-self.points[self.hull.simplices[:, 1]]
        vac = self.points[self.hull.simplices[:, 0]]-self.points[self.hull.simplices[:, 2]]
        nrmls = np.cross(vab, vac)
        self.facet_areas = 0.5*np.abs(np.linalg.norm(nrmls.copy(), axis=1))
        # Scale normal vectors to unit length
        nrmlen = np.sum(nrmls**2, axis=-1)**(1./2)
        nrmls = nrmls*np.tile((1/nrmlen), (3, 1)).T
        # Center of Mass
        center = np.mean(self.points, axis=0)
        # Any point from each simplex
        a = self.points[self.hull.simplices[:, 0]]
        # Make sure all normals point inwards
        dp = np.sum((np.tile(center, (len(a), 1))-a)*nrmls, axis=-1)
        k = dp < 0
        nrmls[k] = -nrmls[k]
        self.normals = nrmls

    def save_hull_as_ply(self, fname):
        header = '\n'.join(['ply','format ascii 1.0','element vertex {}',
                            'property float x', 'property float y',
                            'property float z', 'element face {}',
                            'property list uchar int vertex_indices', 'end_header'])

        header = header.format(self.points.shape[0], self.hull.simplices.shape[0])

        vertices = ''
        for i in range(self.points.shape[0]):
            vertex = '{} {} {}\n'.format(*self.points[i])
            vertices += vertex

        faces = ''
        for simplex in self.hull.simplices:
            face = '3 {} {} {}\n'.format(*simplex)
            faces += face

        ply = header+vertices+faces

        with open(fname, 'w') as f:
            f.write(ply)

    def coordinate_transform(self, R, p):
        """
        Given a 3x3 rotation matrix R and a (3,1) translation vector p, perform
        the coordinate transformation on the point cloud.
        """
        self.points = (R @ self.points.T + p).T

    def coordinate_threshold(self, coord, low, high):
        """
        Remove points that fall outside of range from low to high on
        coord dimension.
        Returns a new point cloud object.

        Coord must be one of 0,1,2 corresponding to x,y,z.
        Low and high are the bounds. All points outside will be discarded.
        """
        low_mask = self.points[:,coord]>low
        high_mask = self.points[:,coord]<high
        pts = self.points[low_mask*high_mask]
        return PointCloud(pts, self.form_hull)

    def box_threshold(self, xlim, ylim, zlim):
        """
        Remove all points not inside a box defined by the iterables xlim, ylim,
        zlim. Returns a new PointCloud object with only the points inside the
        box.

        Each of xlim, ylim, zlim must be iterables of length 2, with the first
        element being the low value and the second being the high value of the
        range.
        """
        low_mask = self.points[:,0]>xlim[0]
        high_mask = self.points[:,0]<xlim[1]
        pts = self.points[low_mask*high_mask]

        low_mask = pts[:,1]>ylim[0]
        high_mask = pts[:,1]<ylim[1]
        pts = pts[low_mask*high_mask]

        low_mask = pts[:,2]>zlim[0]
        high_mask = pts[:,2]<zlim[1]
        pts = pts[low_mask*high_mask]

        return PointCloud(pts, self.form_hull)

    def check_collision(self, points):
        """
        Check if points collides/intersects with the hull of this point cloud.

        points should be an (N,3) ndarray of points to check for collision.
        """
        return self.delaunay.find_simplex(points)>=0

    def get_points(self):
        return self.points

    def back_project(self, R=np.eye(3), p=np.zeros(3), img_shape=(480,640),
                                        x_center=312.307, y_center=245.558,
                                        focal_x=473.852, focal_y=473.852):
        """
        Back project the point cloud into a depth image. The function assumes
        that the camera is positioned at the origin facing along the positive z
        direction.

        The parameters rotation matrix R and vector p can be passed to apply
        coordinate transform so that the camera can be positioned at will. The
        img_shape parameter defines the size of the depth image to be made. The
        remaining parameters define the intrinsic camera parameters of the camera
        being used.

        If only the defaults are used then we assume that the camera is the
        RealSense SR300 and that points are already in the coodinates that place
        the camera at the origin pointing along z (ie camera coordinates)
        """
        points = (R @ self.points.T + p).T
        xs = np.round(points[:,0]*focal_x/points[:,2] + x_center)
        ys = np.round(points[:,1]*focal_y/points[:,2] + y_center)
        depth_img = np.zeros(img_shape)
        for i,(x,y) in enumerate(zip(xs,ys)):
            try:
                depth_img[int(y),int(x)] = points[i,2]
            except IndexError:
                continue
        return DepthImage(depth_img, in_meters=True)
