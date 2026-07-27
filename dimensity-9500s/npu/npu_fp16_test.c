// npu_fp16_test.c — does the MediaTek MDLA do FLOAT16 inference (not just int8)?
// Builds a single fp16 FULLY_CONNECTED and runs it on mtk-mdla.
#include <dlfcn.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>

#define TENSOR_FLOAT16 8
#define INT32          1
#define FULLY_CONNECTED 9
#define NO_ERROR       0
#define M 16
#define K 512
#define N 512

typedef struct { int32_t type; uint32_t dimCount; const uint32_t* dims; float scale; int32_t zero; } OperandType;

#define LOAD(h,n) n=(typeof(n))dlsym(h,#n); if(!n){printf("missing %s\n",#n);return 1;}
#define CK(c) do{int r=(c); if(r!=NO_ERROR){printf("FAIL rc=%d at %s\n",r,#c);return 1;}}while(0)

int main(void){
    void*h=dlopen("libneuronusdk_adapter.mtk.so",RTLD_NOW);
    if(!h){printf("dlopen fail\n");return 1;}
    int(*Neuron_getDeviceCount)(uint32_t*); int(*Neuron_getDevice)(uint32_t,void**);
    int(*NeuronDevice_getName)(const void*,const char**);
    int(*NeuronModel_create)(void**); int(*NeuronModel_addOperand)(void*,const OperandType*);
    int(*NeuronModel_setOperandValue)(void*,int32_t,const void*,size_t);
    int(*NeuronModel_addOperation)(void*,int32_t,uint32_t,const uint32_t*,uint32_t,const uint32_t*);
    int(*NeuronModel_identifyInputsAndOutputs)(void*,uint32_t,const uint32_t*,uint32_t,const uint32_t*);
    int(*NeuronModel_finish)(void*);
    int(*NeuronCompilation_createForDevices)(void*,const void*const*,uint32_t,void**);
    int(*NeuronCompilation_finish)(void*);
    int(*NeuronExecution_create)(void*,void**);
    int(*NeuronExecution_setInput)(void*,int32_t,const void*,const void*,size_t);
    int(*NeuronExecution_setOutput)(void*,int32_t,const void*,void*,size_t);
    int(*NeuronExecution_compute)(void*);
    LOAD(h,Neuron_getDeviceCount)LOAD(h,Neuron_getDevice)LOAD(h,NeuronDevice_getName)
    LOAD(h,NeuronModel_create)LOAD(h,NeuronModel_addOperand)LOAD(h,NeuronModel_setOperandValue)
    LOAD(h,NeuronModel_addOperation)LOAD(h,NeuronModel_identifyInputsAndOutputs)LOAD(h,NeuronModel_finish)
    LOAD(h,NeuronCompilation_createForDevices)LOAD(h,NeuronCompilation_finish)
    LOAD(h,NeuronExecution_create)LOAD(h,NeuronExecution_setInput)LOAD(h,NeuronExecution_setOutput)LOAD(h,NeuronExecution_compute)

    uint32_t nd=0; Neuron_getDeviceCount(&nd); void*dev=0; const char*dn=0;
    for(uint32_t i=0;i<nd;i++){void*d;const char*nm;Neuron_getDevice(i,&d);NeuronDevice_getName(d,&nm);if(strstr(nm,"mtk-mdla")){dev=d;dn=nm;}}
    if(!dev){printf("no mdla\n");return 1;}
    printf("device: %s\n",dn);

    void*model; CK(NeuronModel_create(&model));
    static __fp16 W[N*K], B[N], IN[M*K]; static __fp16 OUT[M*N];
    for(int i=0;i<N*K;i++) W[i]=(__fp16)(((i%7)-3)*0.1f);
    for(int i=0;i<N;i++) B[i]=(__fp16)0.01f;
    for(int i=0;i<M*K;i++) IN[i]=(__fp16)(((i%5)-2)*0.2f);

    uint32_t dIn[2]={M,K}, dW[2]={N,K}, dB[1]={N}, dOut[2]={M,N};
    OperandType tIn={TENSOR_FLOAT16,2,dIn,0,0}, tW={TENSOR_FLOAT16,2,dW,0,0},
                tB={TENSOR_FLOAT16,1,dB,0,0}, tAct={INT32,0,0,0,0}, tOut={TENSOR_FLOAT16,2,dOut,0,0};
    CK(NeuronModel_addOperand(model,&tIn));   //0
    CK(NeuronModel_addOperand(model,&tW));     //1
    CK(NeuronModel_addOperand(model,&tB));     //2
    CK(NeuronModel_addOperand(model,&tAct));   //3
    CK(NeuronModel_addOperand(model,&tOut));   //4
    CK(NeuronModel_setOperandValue(model,1,W,sizeof(W)));
    CK(NeuronModel_setOperandValue(model,2,B,sizeof(B)));
    int32_t act=0; CK(NeuronModel_setOperandValue(model,3,&act,sizeof(act)));
    uint32_t ins[4]={0,1,2,3}, outs[1]={4};
    CK(NeuronModel_addOperation(model,FULLY_CONNECTED,4,ins,1,outs));
    uint32_t mi[1]={0}, mo[1]={4};
    CK(NeuronModel_identifyInputsAndOutputs(model,1,mi,1,mo));
    CK(NeuronModel_finish(model));

    void*comp; const void*devs[1]={dev};
    CK(NeuronCompilation_createForDevices(model,devs,1,&comp));
    CK(NeuronCompilation_finish(comp));
    void*exec; CK(NeuronExecution_create(comp,&exec));
    CK(NeuronExecution_setInput(exec,0,0,IN,sizeof(IN)));
    CK(NeuronExecution_setOutput(exec,0,0,OUT,sizeof(OUT)));
    CK(NeuronExecution_compute(exec));
    printf("out[0]=%.4f out[1]=%.4f\n",(float)OUT[0],(float)OUT[1]);
    printf("RESULT: FLOAT16 inference RUNS on %s\n",dn);
    return 0;
}
