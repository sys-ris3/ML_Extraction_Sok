import logging
import shrub
import tflite
import numpy as np

shrub.util.formatLogging(logging.DEBUG)

def parse_testnet():
    path = 'testnet_obf.tflite'
    with open(path, 'rb') as f:
        buf = f.read()
        model = tflite.Model.GetRootAsModel(buf, 0)
    print("model.Version : %s"%(model.Version()))
    print("model.Description: " + model.Description().decode('utf-8'))
    print("model.SubgraphsLength: %s" %(model.SubgraphsLength()))
    print("model.BuffersLength: %s" % model.BuffersLength())


    graph = model.Subgraphs(0)

    print("graph.InputsLength: %s" % graph.InputsLength())
    print("graph.OutputsLength: %s" % graph.OutputsLength())
    print("graph.InputsAsNumpy: %s" % graph.InputsAsNumpy())
    print("graph.OutputsAsNumpy: %s" % graph.OutputsAsNumpy())

    print("graph.Inputs(0): %s" % (graph.Inputs(0)))
    print("graph.Outputs(0): %s" % graph.Outputs(0))

    print("graph.OperatorsLength(): %s" % (graph.OperatorsLength()))

    op = graph.Operators(0)
    op_code = model.OperatorCodes(op.OpcodeIndex())
    assert(op_code.BuiltinCode() == tflite.BuiltinOperator.CUSTOM)
    #assert(op_code.BuiltinCode() == tflite.BuiltinOperator.CONV_2D)
    print("op.InputsLength: %s" % op.InputsLength())
    print("op.OutputsLength: %s" % op.OutputsLength())

    tensor_index = op.Inputs(1)
    tensor = graph.Tensors(tensor_index)
    print("tensor.ShapeLength %s"%(tensor.ShapeLength()))
    print("tensor.ShapeAsNumpy()[1]: %s " % (tensor.ShapeAsNumpy()[1]))
    # All arrays can dump as Numpy array, or access individually.
    print("tensor.Shape(1): %s" % (tensor.Shape(1)))
    print("tensor.Type(): %s " % (tensor.Type()))
    print("tensor.Name().decode('utf-8'): %s" % (tensor.Name().decode('utf-8')))
    print("tensor.Buffer(): %s" % (tensor.Buffer())) 

    buf = model.Buffers(tensor.Buffer())

    print("buf.DataLength() : %s" % (buf.DataLength()))
    print("buf.DataAsNumpy()[0]: %s "% buf.DataAsNumpy()[0])
    # All arrays can dump as Numpy array, or access individually.
    print("buf.Data(0): %s " % (buf.Data(0)))
    npa = buf.DataAsNumpy()
    print("npa.shape: %s " % (npa.shape))
    
    fh = open("weights.bin", "b+w")
    #write major, minor revision and seen
    major = 0
    minor = 2
    revision = 0 # type int, 4 bytes
    seen = 0 #type size_t, 8bytes
    nda = np.array([major, minor, revision, seen, seen], dtype=np.int32)
    nda.tofile(fh)

    for i in range(graph.OperatorsLength()):
        print("\top id: %s" % i)
        op = graph.Operators(i)
        print("\top.InputsLength: %s" % op.InputsLength())
        print("\top.OutputsLength: %s" % op.OutputsLength())
        if i == 1:
            tensor_index = op.Inputs(2)
            tensor = graph.Tensors(tensor_index)
            buf = model.Buffers(tensor.Buffer())
            npa = buf.DataAsNumpy()
            npa.tofile(fh)
            print("\t\ttensor id: %s" % 2)
            if type(npa)!=int:
                print("\t\ttensor shape: %s" % npa.shape)
                if(len(buf.DataAsNumpy()) > 4):
                    print("\t\tbuf.DataAsNumpy()[0-7]: (%s,%s,%s,%s,%s,%s,%s,%s) "% (buf.DataAsNumpy()[0] , \
                        buf.DataAsNumpy()[1],\
                        buf.DataAsNumpy()[2],\
                        buf.DataAsNumpy()[3],\
                        buf.DataAsNumpy()[4],\
                        buf.DataAsNumpy()[5],\
                        buf.DataAsNumpy()[6],\
                        buf.DataAsNumpy()[7]))
                else:
                    print("\t\tbuf.DataAsNumpy()[0-3]: (%s,%s,%s,%s) "% (buf.DataAsNumpy()[0] , \
                        buf.DataAsNumpy()[1],\
                        buf.DataAsNumpy()[2],\
                        buf.DataAsNumpy()[3]))
            print("")
            tensor_index = op.Inputs(1)
            tensor = graph.Tensors(tensor_index)
            buf = model.Buffers(tensor.Buffer())
            npa = buf.DataAsNumpy()
            npa.tofile(fh)
            print("\t\ttensor id: %s" % 1)
            if type(npa)!=int:
                if(len(buf.DataAsNumpy()) > 4):
                    print("\t\tbuf.DataAsNumpy()[0-7]: (%s,%s,%s,%s,%s,%s,%s,%s) "% (buf.DataAsNumpy()[0] , \
                        buf.DataAsNumpy()[1],\
                        buf.DataAsNumpy()[2],\
                        buf.DataAsNumpy()[3],\
                        buf.DataAsNumpy()[4],\
                        buf.DataAsNumpy()[5],\
                        buf.DataAsNumpy()[6],\
                        buf.DataAsNumpy()[7]))
                else:
                    print("\t\tbuf.DataAsNumpy()[0-3]: (%s,%s,%s,%s) "% (buf.DataAsNumpy()[0] , \
                        buf.DataAsNumpy()[1],\
                        buf.DataAsNumpy()[2],\
                        buf.DataAsNumpy()[3]))
            print("")
            print("")
            continue 

        for j in range(op.InputsLength()):
            print("\t\ttensor id: %s" % j)
            tensor_index = op.Inputs(j)
            tensor = graph.Tensors(tensor_index)
            buf = model.Buffers(tensor.Buffer())
            npa = buf.DataAsNumpy()
            if type(npa)!=int:
                print("\t\ttensor shape: %s" % npa.shape)
                npa.tofile(fh)
                if(len(buf.DataAsNumpy()) > 4):
                    print("\t\tbuf.DataAsNumpy()[0-7]: (%s,%s,%s,%s,%s,%s,%s,%s) "% (buf.DataAsNumpy()[0] , \
                        buf.DataAsNumpy()[1],\
                        buf.DataAsNumpy()[2],\
                        buf.DataAsNumpy()[3],\
                        buf.DataAsNumpy()[4],\
                        buf.DataAsNumpy()[5],\
                        buf.DataAsNumpy()[6],\
                        buf.DataAsNumpy()[7]))
                else:
                    print("\t\tbuf.DataAsNumpy()[0-3]: (%s,%s,%s,%s) "% (buf.DataAsNumpy()[0] , \
                        buf.DataAsNumpy()[1],\
                        buf.DataAsNumpy()[2],\
                        buf.DataAsNumpy()[3]))

            else:
                print("\t\tnpa is int : %s" %npa)
            print("\t\tbuf.DataLength() : %s" % (buf.DataLength()))
            print("")
        print("")
        print("")


if __name__ == '__main__':
    parse_testnet()
