#!/usr/bin/python3
from URDFParser import URDFParser
from RBDReference import RBDReference
from GRiDCodeGenerator import GRiDCodeGenerator
from util import parseInputs, printUsage, validateRobot, initializeValues, printErr
import numpy as np

def main():
    URDF_PATH, DEBUG_MODE, FILE_NAMESPACE_NAME = parseInputs()

    parser = URDFParser()
    robot = parser.parse(URDF_PATH)

    validateRobot(robot)

    reference = RBDReference(robot)
    q, qd, u, n = initializeValues(robot, MATCH_CPP_RANDOM = True)

    print("q\n",q)
    print("qd\n",qd)
    print("u\n",u)

    ee_pos = reference.end_effector_positions(q)
    print("eepos\n",ee_pos)

    dee_pos = reference.end_effector_position_gradients(q)
    print("deepos\n",dee_pos)

    (c, v, a, f) = reference.rnea(q,qd)
    print("c\n",c)

    Minv = reference.minv(q)
    print("Minv\n", Minv)

    qdd = np.matmul(Minv,(u-c))
    print("qdd\n",qdd)

    crba=reference.crba(q,qd,u)
    print("crba\n",crba)

    qdd_aba = reference.aba(q,qd,u)
    print("aba\n",qdd_aba)
    
    dc_du = reference.rnea_grad(q, qd, qdd)
    print("dc/dq with qdd\n",dc_du[:,:n])
    print("dc/dqd with qdd\n",dc_du[:,n:])

    df_du = np.matmul(-Minv,dc_du)
    print("df/dq\n",df_du[:,:n])
    print("df/dqd\n",df_du[:,n:])

    if DEBUG_MODE:
        print("-------------------")
        print("printing intermediate outputs from refactorings")
        print("-------------------")
        codegen = GRiDCodeGenerator(robot, DEBUG_MODE, FILE_NAMESPACE = FILE_NAMESPACE_NAME)
        (c, v, a, f) = codegen.test_rnea(q,qd)
        print("v\n",v)
        print("a\n",a)
        print("f\n",f)
        print("c\n",c)
        
        Minv = codegen.test_minv(q)
        print("Minv\n",Minv)

        umc = u-c
        print("u-c\n",umc)

        qdd = np.matmul(Minv,umc)
        print("qdd\n",qdd)
        
        dc_du = codegen.test_rnea_grad(q, qd, qdd)
        print("dc/dq with qdd\n",dc_du[:,:n])
        print("dc/dqd with qdd\n",dc_du[:,n:])
        
        df_du = np.matmul(-Minv,dc_du)
        print("df/dq\n",df_du[:,:n])
        print("df/dqd\n",df_du[:,n:])

if __name__ == "__main__":
    main()