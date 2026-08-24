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
    private int ctx = 2048, maxTok = 256, topK = 40, threads = 0;
    private float temp = 0.7f;

    @Override protected void onCreate(Bundle b) {
        super.onCreate(b);
        setContentView(R.layout.activity_chat);
        out = findViewById(R.id.out);
        scroll = findViewById(R.id.scroll);
        input = findViewById(R.id.input);
        send = findViewById(R.id.send);
        busy = findViewById(R.id.busy);
        findViewById(R.id.back).setOnClickListener(v -> finish());

        SharedPreferences sp = getSharedPreferences("settings", MODE_PRIVATE);
        ctx    = sp.getInt("ctx", 2048);
        maxTok = sp.getInt("maxtok", 256);
        temp   = sp.getFloat("temp", 0.7f);
        topK   = sp.getInt("topk", 40);
        threads= sp.getInt("threads", 0);

        String path = getIntent().getStringExtra("model");
        ((TextView) findViewById(R.id.title)).setText(new File(path).getName());
        Native.setThreads(threads);

        new Thread(() -> {
            final long ptr = Native.loadModel(path, ctx);
            runOnUiThread(() -> {
                if (ptr == 0) { finish(); return; }
                model = ptr;
                out.setText(String.format(
                    "✓ Modelo cargado · ctx=%d\nEscribe tu mensaje.", ctx));
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
                out.append("\n\n\u25AA Tú: " + q + "\n\u25AA Bot: ");
                scroll.post(() -> scroll.fullScroll(View.FOCUS_DOWN));
            });
            new Thread(() -> {
                long t0 = System.currentTimeMillis();
                final String reply = Native.generate(model, history.toString(), maxTok, temp, topK,
                    t -> ui.post(() -> {
                        out.append(t);
                        scroll.post(() -> scroll.fullScroll(View.FOCUS_DOWN));
                    }));
                if (reply != null) history.append(reply);
                final long ms = System.currentTimeMillis() - t0;
                ui.post(() -> {
                    send.setEnabled(true); input.setEnabled(false ? true : true);
                    input.setEnabled(true); busy.setVisibility(View.GONE);
                    out.append(String.format("\n\n[%d tok en %.1fs]", maxTok, ms / 1000.0));
                    scroll.post(() -> scroll.fullScroll(View.FOCUS_DOWN));
                });
            }).start();
        });
    }

    /* evita desbordar el contexto: recorta el historial a ~3/4 del mismo */
    private void trimHistory() {
        int approxTokens = history.length() / 3;   /* ~3 caracteres por token */
        int limit = ctx * 3 / 4;
        while (approxTokens > limit && history.length() > 256) {
            int cut = history.indexOf("\nUsuario:", 16);
            if (cut < 0) break;
            history.delete(0, cut);
            approxTokens = history.length() / 3;
        }
    }

    @Override protected void onDestroy() {
        super.onDestroy();
        if (model != 0) { Native.freeModel(model); model = 0; }
    }
}
