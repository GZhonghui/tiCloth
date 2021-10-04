import taichi as ti

ti.init(arch=ti.cuda)

Mu = 0.2

massLength = 0.1
massK      = 1000.0

pointMass = 0.2

widthSize,heightSize = 63, 63

pointSize = (widthSize + 1) * (heightSize + 1)

pointLocation = ti.Vector.field(3, dtype=ti.f32, shape=pointSize)
pointForce    = ti.Vector.field(3, dtype=ti.f32, shape=pointSize)

G = ti.Vector([0.0, 0.0, -9.8], dt=ti.f32)

@ti.func
def pointID(x,y):
  R = -1
  if 0 <= x and x <= widthSize and 0 <= y and y <= heightSize:
    R = y * (widthSize + 1) + x
  return R

@ti.func
def pointCoord(ID):
  return (ID % (widthSize + 1), ID // (widthSize + 1))

@ti.func
def massID(ID):
  R = ti.Vector([-1, -1, -1, -1], dt=ti.i32)
  x,y = pointCoord(ID)
  R[0],R[1] = pointID(x-1, y),pointID(x+1, y)
  R[2],R[3] = pointID(x, y-1),pointID(x, y+1)
  return R

@ti.kernel
def Init():
  for i in pointLocation:
    x,y = pointCoord(i)
    pointLocation[i] = (x * massLength, y * massLength, 6)

@ti.kernel
def ComputeForce():
  for i in pointForce:
    pointForce[i] = G * pointMass
    fourDir = massID(i)
    for j in ti.static(range(4)):
      if not fourDir[j] == -1:
        Dir = pointLocation[fourDir[j]] - pointLocation[i]
        pointForce[i] += (Dir.norm() - massLength) * massK * Dir / Dir.norm()

@ti.kernel
def Forward(T: ti.f32):
  for i in pointLocation:
    x,y = pointCoord(i)
    if x:
      V = pointForce[i] / pointMass
      pointLocation[i] += T * V

@ti.kernel
def ComputeColl():
  pass

def Step():
  for i in range(30):
    ComputeForce()
    Forward(0.00001)

def Export(i: int):
  npL = pointLocation.to_numpy()

  mesh_writer = ti.PLYWriter(num_vertices=pointSize, face_type="quad")
  mesh_writer.add_vertex_pos(npL[:, 0], npL[:, 1], npL[:, 2])

  mesh_writer.export_frame_ascii(i, 'S.ply')

  print('Frame >> %03d'%(i))

def main():
  Init()
  Frame = 0
  try:
    while True:
      Step()

      Frame += 1
      if not Frame % 50:
        Export(Frame // 50)
  except Exception as Error:
    print(Error)

if __name__=='__main__':
  main()