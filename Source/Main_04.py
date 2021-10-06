# Ref: https://graphics.stanford.edu/~mdfisher/cloth.html

import taichi as ti

ti.init(arch=ti.cuda)

# Air resistance
A = 0.01

massLengthA = 0.1
massLengthB = ti.sqrt(2 * massLengthA * massLengthA)
massK      = 50000.0

pointMass = 0.2

widthSize,heightSize = 127, 127

faceSize = widthSize * heightSize * 2

pointSize = (widthSize + 1) * (heightSize + 1)

pointLocation = ti.Vector.field(3, dtype=ti.f32, shape=pointSize)
pointVelocity = ti.Vector.field(3, dtype=ti.f32, shape=pointSize)
pointForce    = ti.Vector.field(3, dtype=ti.f32, shape=pointSize)

Idx = ti.Vector.field(3, dtype=ti.i32, shape=faceSize)
vUV = ti.Vector.field(2, dtype=ti.f32, shape=pointSize)

# Y Forward
G = ti.Vector([0.0, 0.0, -9.8], dt=ti.f32)

Wind = ti.Vector([0.3, 0.0, 0.0], dt=ti.f32)

@ti.func
def pointID(x,y):
  R = -1
  if 0 <= x and x <= widthSize and 0 <= y and y <= heightSize:
    R = y * (widthSize + 1) + x
  return R

def pointIDPy(x,y):
  R = -1
  if 0 <= x and x <= widthSize and 0 <= y and y <= heightSize:
    R = y * (widthSize + 1) + x
  return R

@ti.func
def pointCoord(ID):
  return (ID % (widthSize + 1), ID // (widthSize + 1))

@ti.func
def massID(ID):
  R = ti.Vector([-1, -1, -1, -1, -1, -1, -1, -1], dt=ti.i32)
  x,y = pointCoord(ID)
  R[0],R[1] = pointID(x-1, y),pointID(x+1, y)
  R[2],R[3] = pointID(x, y-1),pointID(x, y+1)
  R[4],R[5] = pointID(x-1,y-1),pointID(x+1,y+1)
  R[6],R[7] = pointID(x-1,y+1),pointID(x+1,y-1)
  return R

@ti.kernel
def InitTi():
  for i in range(pointSize):
    x,y = pointCoord(i)
    pointLocation[i] = (0, y * massLengthA, 10 - x * massLengthA)
    pointVelocity[i] = (0, 0, 0)
    vUV[i][1] = 1.0 - ti.cast(x, ti.f32) / widthSize
    vUV[i][0] = ti.cast(y, ti.f32) / heightSize

@ti.kernel
def ComputeForce():
  for i in pointForce:
    pointForce[i] = (0, 0, 0)
    Dirs = massID(i)
    for j in ti.static(range(0,4)):
      if not Dirs[j] == -1:
        Dir = pointLocation[Dirs[j]] - pointLocation[i]
        pointForce[i] += (Dir.norm() - massLengthA) * massK * Dir / Dir.norm()
    for j in ti.static(range(4,8)):
      if not Dirs[j] == -1:
        Dir = pointLocation[Dirs[j]] - pointLocation[i]
        pointForce[i] += (Dir.norm() - massLengthB) * massK * Dir / Dir.norm()
    pointForce[i] += G * pointMass + Wind
    pointForce[i] += A * pointVelocity[i] * pointVelocity[i]
    x,y = pointCoord(i)
    if not x:
        pointForce[i] = (0, 0, 0)

@ti.kernel
def Forward(T: ti.f32, SumT: ti.f32):
  for i in range(pointSize):
    pointVelocity[i] += T * pointForce[i] / pointMass
    pointLocation[i] += T * pointVelocity[i]
    x,y = pointCoord(i)
    if not x:
        Angle = min(SumT, 3.1415926 * 2)
        cP = ti.Vector([0.0, 0.5 * heightSize * massLengthA])
        dX = 0.0 - cP[0]
        dY = y * massLengthA - cP[1]
        pointLocation[i][0] = dX * ti.cos(Angle) - dY * ti.sin(Angle) + cP[0]
        pointLocation[i][1] = dY * ti.cos(Angle) + dX * ti.sin(Angle) + cP[1]

def Init():
  InitTi()

  Index = 0
  for i in range(widthSize):
    for j in range(heightSize):
      ID_1 = pointIDPy(i,j)
      ID_2 = pointIDPy(i+1,j)
      ID_3 = pointIDPy(i,j+1)
      ID_4 = pointIDPy(i+1,j+1)

      Idx[Index + 0][0] = ID_1
      Idx[Index + 0][1] = ID_2
      Idx[Index + 0][2] = ID_3
      Idx[Index + 1][0] = ID_4
      Idx[Index + 1][1] = ID_3
      Idx[Index + 1][2] = ID_2

      Index += 2

def Step(SumT):
  for i in range(50):
    ComputeForce()
    Forward(1e-5, SumT)
    SumT += 1e-5
  return SumT

def Export(frameIndex: int):
  npL = pointLocation.to_numpy()
  npI = Idx.to_numpy()
  npU = vUV.to_numpy()

  fileName = 'S_%03d.obj'%(frameIndex)
  with open(fileName, 'w') as F:
    for i in range(pointSize):
      F.write('v %.4f %.4f %.4f\n'%(npL[i,0],npL[i,1],npL[i,2]))
    for i in range(pointSize):
      F.write('vt %.4f %.4f\n'%(npU[i,0],npU[i,1]))
    for i in range(faceSize):
      x,y,z = npI[i,0]+1,npI[i,1]+1,npI[i,2]+1
      F.write('f %d/%d %d/%d %d/%d\n'%(x,x,y,y,z,z))

  print('Frame >> %03d'%(frameIndex))

def main():
  Init()
  Frame = 0
  SumT = 0.0
  try:
    while True:
      SumT = Step(SumT)

      Frame += 1
      if not Frame % 60:
        Export(Frame // 60)
  except Exception as Error:
    print(Error)

if __name__=='__main__':
  main()