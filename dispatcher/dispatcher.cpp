// CPU の SIMD を判定し engine\YaneuraOu-<ISA>.exe を透過起動する USI ランチャ。
// stdin/stdout/stderr ハンドルを子に継承させ、GUI と子が直接 USI 通信する
// (ディスパッチャはプロトコルに介入しない = 透過)。
// GetModuleFileNameW / CreateProcessW と wide API を使うため、配置パスが非ASCIIでも動く。
#include <windows.h>
#include <string>
#include <vector>
#include <cstdio>

#if defined(__GNUC__)
#include <cpuid.h>
static void cpuidex(unsigned leaf, unsigned sub, unsigned r[4]) {
    __cpuid_count(leaf, sub, r[0], r[1], r[2], r[3]);
}
static unsigned long long xgetbv0() {
    unsigned int a, d;
    __asm__ __volatile__("xgetbv" : "=a"(a), "=d"(d) : "c"(0));
    return ((unsigned long long)d << 32) | a;
}
#else
#include <intrin.h>
static void cpuidex(unsigned leaf, unsigned sub, unsigned r[4]) { __cpuidex((int*)r, (int)leaf, (int)sub); }
static unsigned long long xgetbv0() { return _xgetbv(0); }
#endif

// 利用可能な最良の ISA を返す。優先 AVX512 > AVX2 > SSE42。
static const wchar_t* detect_isa() {
    unsigned r[4];
    cpuidex(1, 0, r);
    bool osxsave = (r[2] >> 27) & 1;
    bool avx     = (r[2] >> 28) & 1;
    unsigned long long xcr0 = osxsave ? xgetbv0() : 0ull;
    bool os_ymm = ((xcr0 & 0x6) == 0x6);      // XMM + YMM state 保存が OS 有効
    bool os_zmm = ((xcr0 & 0xE6) == 0xE6);    // + opmask + ZMM_hi256 + Hi16_ZMM
    cpuidex(7, 0, r);
    bool avx2    = (r[1] >> 5) & 1;
    bool avx512f = (r[1] >> 16) & 1;
    if (avx512f && avx && os_ymm && os_zmm) return L"AVX512";
    if (avx2 && avx && os_ymm)              return L"AVX2";
    return L"SSE42"; // フォールバック
}

// 実行中バイナリのフォルダ (末尾に \ 付き) を wide で返す。
static std::wstring self_dir() {
    std::wstring buf(32768, L'\0');
    DWORD n = GetModuleFileNameW(nullptr, &buf[0], (DWORD)buf.size());
    buf.resize(n);
    size_t pos = buf.find_last_of(L"\\/");
    return (pos == std::wstring::npos) ? std::wstring(L".\\") : buf.substr(0, pos + 1);
}

// GetCommandLineW から自プログラム名 (argv[0]) を除いた残りを返す。
static std::wstring args_after_argv0() {
    std::wstring s(GetCommandLineW());
    size_t i = 0;
    if (i < s.size() && s[i] == L'"') {          // "quoted program"
        i++;
        while (i < s.size() && s[i] != L'"') i++;
        if (i < s.size()) i++;                    // 閉じ引用符
    } else {
        while (i < s.size() && s[i] != L' ' && s[i] != L'\t') i++;
    }
    while (i < s.size() && (s[i] == L' ' || s[i] == L'\t')) i++;
    return s.substr(i);
}

int main() {
    const wchar_t* isa = detect_isa();
    std::wstring child = self_dir() + L"engine\\YaneuraOu-" + isa + L".exe";
    // 診断は stderr のみ (stdout は子専用。USI プロトコルを汚さない)。
    fwprintf(stderr, L"[dispatcher] selected ISA=%ls -> %ls\n", isa, child.c_str());
    fflush(stderr);

    std::wstring rest = args_after_argv0();
    std::wstring cmdline = L"\"" + child + L"\"";
    if (!rest.empty()) cmdline += L" " + rest;

    STARTUPINFOW si{}; si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdInput  = GetStdHandle(STD_INPUT_HANDLE);
    si.hStdOutput = GetStdHandle(STD_OUTPUT_HANDLE);
    si.hStdError  = GetStdHandle(STD_ERROR_HANDLE);
    PROCESS_INFORMATION pi{};
    std::vector<wchar_t> buf(cmdline.begin(), cmdline.end()); buf.push_back(L'\0');

    if (!CreateProcessW(child.c_str(), buf.data(), nullptr, nullptr, TRUE, 0,
                        nullptr, nullptr, &si, &pi)) {
        fwprintf(stderr, L"[dispatcher] failed to launch %ls (err=%lu)\n", child.c_str(), GetLastError());
        return 1;
    }
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD code = 1; GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    return (int)code;
}
