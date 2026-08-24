# build_apk.ps1 — compila gguf2bin.apk sin Gradle usando NDK + JDK + build-tools
$ErrorActionPreference = "Stop"
$sdk    = "C:\Android\sdk"
$ndk    = "$sdk\ndk\28.2.13676358"
$bt     = "$sdk\build-tools\35.0.0"
$plat   = "$sdk\platforms\android-35\android.jar"
$root   = $PSScriptRoot
$out    = "$root\build"
New-Item -ItemType Directory -Force "$out" | Out-Null

# 0) debug keystore (una vez)
$ks = "$out\debug.keystore"
if (!(Test-Path $ks)) {
  keytool -genkeypair -keystore $ks -alias androiddebugkey -storepass android `
    -keypass android -dname "CN=Android Debug,O=Android,C=US" `
    -keyalg RSA -keysize 2048 -validity 10000
}

# 1) núcleo C -> libgguf2bin.so (arm64)
$cc = "$ndk\toolchains\llvm\prebuilt\windows-x86_64\bin\aarch64-linux-android24-clang.cmd"
$srcs = @("$root\..\src\l1_gguf.c","$root\..\src\l2_codec.c","$root\..\src\l3_math.c",
          "$root\..\src\l4_gbin.c","$root\..\src\l5_model.c","$root\..\src\l6_token.c",
          "$root\jni\g2b_jni.c")
& $cc -O3 -ffast-math -fPIC -shared -o "$out\libgguf2bin.so" `
  -I"$root\..\include" @srcs -lm -fopenmp -static-openmp
if ($LASTEXITCODE) { throw "fallo compilación nativa" }

# 2) recursos + manifest -> apk base (+ R.java)
New-Item -ItemType Directory -Force "$out\res_flat" | Out-Null
& "$bt\aapt2.exe" compile --dir "$root\res" -o "$out\res_flat"
if ($LASTEXITCODE) { throw "fallo aapt2 compile" }
& "$bt\aapt2.exe" link -o "$out\base.apk" -I $plat --manifest "$root\AndroidManifest.xml" `
  --java "$root\java" (Get-ChildItem "$out\res_flat" -File).FullName --auto-add-overlay
if ($LASTEXITCODE) { throw "fallo aapt2 link" }

# 3) java -> dex
New-Item -ItemType Directory -Force "$out\classes" | Out-Null
javac --release 8 -cp $plat -d "$out\classes" (Get-ChildItem "$root\java" -Filter *.java -Recurse).FullName
if ($LASTEXITCODE) { throw "fallo javac" }
& "$bt\d8.bat" --release --lib $plat --output "$out" (Get-ChildItem "$out\classes" -Filter *.class -Recurse).FullName
if ($LASTEXITCODE) { throw "fallo d8" }

# 4) meter dex + .so al apk
Push-Location $out
jar uf base.apk classes.dex
New-Item -ItemType Directory -Force "lib\arm64-v8a" | Out-Null
Copy-Item libgguf2bin.so -Destination "lib\arm64-v8a\" -Force
jar uf base.apk lib/arm64-v8a/libgguf2bin.so
Pop-Location

# 5) alinear + firmar
& "$bt\zipalign.exe" -f -P 16 4 "$out\base.apk" "$out\aligned.apk"
if ($LASTEXITCODE) { throw "fallo zipalign" }
& "$bt\apksigner.bat" sign --ks $ks --ks-pass pass:android --key-pass pass:android `
  --out "$root\..\gguf2bin.apk" "$out\aligned.apk"
if ($LASTEXITCODE) { throw "fallo firma" }

Write-Host "`nOK -> gguf2bin.apk" -ForegroundColor Green
