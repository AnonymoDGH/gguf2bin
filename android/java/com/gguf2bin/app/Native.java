package com.gguf2bin.app;

public class Native {
    static { System.loadLibrary("gguf2bin"); }

    public interface TokenSink { void onToken(String s); }

    public static native long loadModel(String path, int ctx);
    public static native void freeModel(long ptr);
    public static native void setThreads(int n);
    public static native String generate(long ptr, String prompt, int maxTokens,
                                         float temp, int topK, TokenSink sink);
}
