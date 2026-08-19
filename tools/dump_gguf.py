import struct, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

GGUF_MAGIC = 0x46554747
GT = {0:'F32',1:'F16',2:'Q4_0',3:'Q4_1',6:'Q5_0',7:'Q5_1',8:'Q8_0',9:'Q8_1',
      10:'Q2_K',11:'Q3_K',12:'Q4_K',13:'Q5_K',14:'Q6_K',15:'Q8_K'}
BS = {0:1,1:1,2:32,3:32,6:32,7:32,8:32,9:32}
BY = {0:4,1:2,2:18,3:20,6:22,7:24,8:34,9:36}

def ru(f, fmt):
    return struct.unpack('<'+fmt, f.read(struct.calcsize('<'+fmt)))[0]

def main(path):
    f = open(path,'rb')
    magic = ru(f,'I'); ver = ru(f,'I')
    n_tensors = ru(f,'Q'); meta_n = ru(f,'Q')
    print(f'magic={magic:#x} ver={ver} tensors={n_tensors} meta={meta_n}')
    alignment = 32
    meta = {}
    for _ in range(meta_n):
        klen = ru(f,'Q'); key = f.read(klen).decode('utf-8','replace')
        vt = ru(f,'I')
        def skip_str():
            n = ru(f,'Q'); f.seek(n,1)
        def read_str():
            n = ru(f,'Q'); return f.read(n).decode('utf-8','replace')
        if vt==0: meta[key]=ru(f,'B')
        elif vt==1: meta[key]=ru(f,'b')
        elif vt==2: meta[key]=ru(f,'H')
        elif vt==3: meta[key]=ru(f,'h')
        elif vt==4: meta[key]=ru(f,'I')
        elif vt==5: meta[key]=ru(f,'i')
        elif vt==6: meta[key]=ru(f,'f')
        elif vt==7: meta[key]=bool(ru(f,'B'))
        elif vt==8: meta[key]=read_str()
        elif vt==9:
            at=ru(f,'I'); n=ru(f,'Q')
            if at==8:
                arr=[read_str() for _ in range(n)]; meta[key]=arr
            else:
                f.seek(struct.calcsize('<8Q' if at in (10,11) else '<f')*0,1)
                sz=8 if at in (10,11,12) else (4 if at in (4,5,6) else (2 if at in (2,3) else 1))
                f.seek(n*sz,1); meta[key]=f'<array {n} x vt{at}>'
        elif vt==10: meta[key]=ru(f,'Q')
        elif vt==11: meta[key]=ru(f,'q')
        elif vt==12: meta[key]=ru(f,'d')
        if key=='general.alignment': alignment=meta[key]
    print('--- metadata ---')
    for k,v in meta.items():
        s=str(v)
        if len(s)>120: s=s[:120]+'...'
        print(f'  {k} = {s}')
    print('--- tensors ---')
    tens=[]
    for _ in range(n_tensors):
        nlen=ru(f,'Q'); name=f.read(nlen).decode('utf-8','replace')
        nd=ru(f,'I'); dims=[ru(f,'Q') for _ in range(nd)]
        ty=ru(f,'I'); off=ru(f,'Q')
        tens.append((name,dims,ty,off))
        ne=1
        for d in dims: ne*=d
        nbytes=(ne//BS.get(ty,1))*BY.get(ty,4)
        print(f'  {name:45s} {GT.get(ty,str(ty)):5s} dims={dims} nbytes={nbytes}')
    f.close()

if __name__=='__main__':
    main(sys.argv[1])
