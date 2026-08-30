package com.gguf2bin.app;

import android.app.Activity;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ProgressBar;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;
import java.io.File;

public class ChatActivity extends Activity {
    private long model = 0;
    private TextView out;
    private ScrollView scroll;
    private EditText input;
    private Button send;
    private ProgressBar busy;
    private final Handler ui = new Handler(Looper.getMainLooper());
    private final StringBuilder history = new StringBuilder();
    private int ctx = 2048, maxTok = 256, topK = 40;
    private float temp = 0.7f;
    private boolean userScrolling = false;
    private volatile boolean closed = false;
    private Thread genThread;

    @Override protected void onCreate(Bundle b) {
        super.onCreate(b);
        setContentView(R.layout.activity_chat);
        out = findViewById(R.id.out);
        scroll = findViewById(R.id.scroll);
        input = findViewById(R.id.input);
        send = findViewById(R.id.send);
        busy = findViewById(R.id.busy);
        findViewById(R.id.back).setOnClickListener(v -> finish());
        send.setEnabled(false);

        SharedPreferences sp = getSharedPreferences("settings", MODE_PRIVATE);
        ctx    = sp.getInt("ctx", 2048);
        maxTok = sp.getInt("maxtok", 256);
        temp   = sp.getFloat("temp", 0.7f);
        topK   = sp.getInt("topk", 40);
        if (scroll.getViewTreeObserver() != null) {
            scroll.getViewTreeObserver().addOnScrollChangedListener(() -> {
                View child = scroll.getChildAt(0);
                if (child == null) return;
                int diff = child.getBottom() - (scroll.getHeight() + scroll.getScrollY());
                userScrolling = diff > 120;
            });
        }

        String path = getIntent().getStringExtra("model");
        if (path == null || !(new File(path).exists())) {
            Toast.makeText(this, "Modelo no encontrado", 1).show();
            finish(); return;
        }
        ((TextView) findViewById(R.id.title)).setText(new File(path).getName());
        out.setText("Cargando modelo...");

        new Thread(() -> {
            long t0 = System.currentTimeMillis();
            final long ptr = Native.loadModel(path, ctx);
            runOnUiThread(() -> {
                if (closed || isFinishing()) {
                    if (ptr != 0) Native.freeModel(ptr);
                    return;
                }
                if (ptr == 0) {
                    Toast.makeText(this, "No se pudo cargar el modelo", 1).show();
                    out.setText("Error al cargar el modelo.");
                    return;
                }
                model = ptr;
                long ms = System.currentTimeMillis() - t0;
                out.setText(String.format("Modelo cargado en %.1fs  ·  ctx=%d\nEscribe tu mensaje abajo.", ms/1000.0, ctx));
                send.setEnabled(true);
            });
        }).start();

        send.setOnClickListener(v -> {
            String q = input.getText().toString().trim();
            if (q.isEmpty() || model == 0) return;
            input.setText("");
            send.setEnabled(false); busy.setVisibility(View.VISIBLE); input.setEnabled(false);
            history.append("\nUsuario: ").append(q).append("\nAsistente: ");
            trimHistory();
            ui.post(() -> {
                out.append("\n\n\u25AA Tu: " + q + "\n\u25AA Bot: ");
                autoScroll();
            });
            genThread = new Thread(() -> {
                final String reply = Native.generate(model, history.toString(), maxTok, temp, topK,
                    t -> ui.post(() -> { if (!closed) { out.append(t); autoScroll(); } }));
                if (reply != null && !closed) history.append(reply);
                ui.post(() -> {
                    if (closed) return;
                    send.setEnabled(true); input.setEnabled(true); busy.setVisibility(View.GONE);
                    autoScroll();
                });
            });
            genThread.start();
        });
    }

    private void autoScroll() {
        if (userScrolling) return;
        scroll.post(() -> scroll.fullScroll(View.FOCUS_DOWN));
    }

    private void trimHistory() {
        int approxTokens = history.length() / 3;
        int limit = ctx * 3 / 4;
        while (approxTokens > limit && history.length() > 256) {
            int cut = history.indexOf("\nUsuario:", 16);
            if (cut < 0) break;
            history.delete(0, cut);
            approxTokens = history.length() / 3;
        }
    }

    @Override protected void onDestroy() {
        closed = true;
        if (model != 0) Native.requestCancel(model);
        if (genThread != null) {
            try { genThread.join(2000); } catch (InterruptedException ignored) {}
        }
        if (model != 0) { Native.freeModel(model); model = 0; }
        super.onDestroy();
    }
}
