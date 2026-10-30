//----------------------------------------------------------------------------------
//	AIオブジェクトマスク 汎用プラグイン (.aux2) for AviUtl ExEdit2
//
//	・フィルタ効果「AIオブジェクトマスク」 ... 指定した枠の中の物体をSAM2で検出し、それ以外を透過する
//	・独自ウィンドウ「AIオブジェクトマスク」 ... 現在のフレームを表示し、ドラッグで枠を指定する
//
//	配置:
//		Plugin\ObjectMask\ObjectMask.aux2
//		Plugin\ObjectMask\onnxruntime.dll (ほか ONNX Runtime のDLL)
//		Plugin\ObjectMask\models\sam2_encoder.onnx
//		Plugin\ObjectMask\models\sam2_decoder.onnx
//----------------------------------------------------------------------------------
#define NOMINMAX
#define _CRT_SECURE_NO_WARNINGS
#include <windows.h>
#include <windowsx.h>
#include <algorithm>
#include <atomic>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <array>
#include <vector>

#include "plugin2.h"
#include "filter2.h"
#include "logger2.h"
#include "config2.h"

#include <cstdlib>

// onnxruntime.dll は、プラグインのフォルダにあるものを自分で読み込んで使う
// (他のプラグインや Windows が先に読み込んだ別バージョンのDLLと混ざってクラッシュするのを避けるため)
#define ORT_API_MANUAL_INIT
#include <onnxruntime_cxx_api.h>
#if defined(__has_include)
#if __has_include(<dml_provider_factory.h>)
#include <dml_provider_factory.h>
#if ORT_API_VERSION >= 13
#define OM_HAS_DML 1
#endif
#endif
#endif

//---------------------------------------------------------------------
//	名称定義 (フィルタの設定項目名とウィンドウからの書き込みで同じ名前を使う)
//---------------------------------------------------------------------
#define FILTER_NAME		L"AIオブジェクトマスク(ObjectMask)"
#define ITEM_BOX_X		L"枠X"
#define ITEM_BOX_Y		L"枠Y"
#define ITEM_BOX_W		L"枠幅"
#define ITEM_BOX_H		L"枠高さ"
#define ITEM_ANCHOR		L"基準フレーム"
#define ITEM_SHAPE		L"選択範囲(自動設定)"

static const int MASK_MAX_SIZE = 512;		// マスクを保持する最大解像度(長辺)
static const int PREVIEW_MAX_WIDTH = 960;	// 枠選択ウィンドウに渡すプレビューの最大幅
static const int TRACK_SEARCH_FRAMES = 30;	// 追跡の起点にするキャッシュ済みフレームの探索範囲
static const size_t CACHE_MAX_FRAMES = 8000;
static const int TRACK_THUMB_W = 200;		// 追跡用サムネイルの幅

//---------------------------------------------------------------------
//	グローバル
//---------------------------------------------------------------------
static LOG_HANDLE* g_logger = nullptr;
static CONFIG_HANDLE* g_config = nullptr;
static EDIT_HANDLE* g_edit_handle = nullptr;
static std::atomic<int> g_generation{ 0 };	// 増やすと全キャッシュと読み込み失敗状態を破棄する

static void log_info(const std::wstring& message)
{
	if (g_logger)
	{
		g_logger->info(g_logger, message.c_str());
	}
}

static void log_warn(const std::wstring& message)
{
	if (g_logger)
	{
		g_logger->warn(g_logger, message.c_str());
	}
}

static void log_error(const std::wstring& message)
{
	if (g_logger)
	{
		g_logger->error(g_logger, message.c_str());
	}
}

static std::wstring widen(const char* utf8)
{
	if (!utf8 || !*utf8)
	{
		return L"";
	}
	int n = MultiByteToWideChar(CP_UTF8, 0, utf8, -1, nullptr, 0);
	if (n <= 0)
	{
		return L"";
	}
	std::wstring s(n, L'\0');
	MultiByteToWideChar(CP_UTF8, 0, utf8, -1, &s[0], n);
	s.resize(n - 1);
	return s;
}

// 行などの範囲を分割して複数スレッドで処理する。f(begin, end)
template <class F>
static void parallel_for(int n, F f)
{
	static const unsigned hw = std::max(1u, std::min(8u, std::thread::hardware_concurrency()));
	int threads = (int)std::min<unsigned>(hw, (unsigned)std::max(1, n / 16));
	if (threads <= 1)
	{
		f(0, n);
		return;
	}
	int chunk = (n + threads - 1) / threads;
	std::vector<std::thread> pool;
	for (int t = 1; t < threads; t++)
	{
		int b = t * chunk;
		int e = std::min(n, (t + 1) * chunk);
		if (b >= e)
		{
			break;
		}
		pool.emplace_back([&f, b, e]() { f(b, e); });
	}
	f(0, std::min(n, chunk));
	for (auto& th : pool)
	{
		th.join();
	}
}

// 表示用に文字列を短くする
static std::wstring short_text(const std::wstring& s, size_t n)
{
	return s.size() <= n ? s : s.substr(0, n) + L"...";
}

static std::wstring plugin_dir()
{
	wchar_t path[MAX_PATH] = {};
	HMODULE mod = nullptr;
	GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, (LPCWSTR)&plugin_dir, &mod);
	GetModuleFileNameW(mod, path, MAX_PATH);
	std::wstring s = path;
	size_t p = s.find_last_of(L"\\/");
	return (p == std::wstring::npos) ? s : s.substr(0, p);
}

//---------------------------------------------------------------------
//	基本型
//---------------------------------------------------------------------
struct BoxF
{
	float x0 = 0, y0 = 0, x1 = 0, y1 = 0;
};

struct LoadError
{
	std::wstring msg;
};

// SAM2 に渡すプロンプト (枠 + 物体上の点 + 背景上の点)
struct PromptSet
{
	BoxF box;
	std::vector<float> pts;		// x,y の並び (元画像座標)
	std::vector<float> labels;	// 点ごとのラベル (1 = 物体, 0 = 背景)
};

typedef std::vector<std::pair<int, int>> PolyPoints;

// 指定した範囲の1つ分。subtract が true なら、そこを対象から除く (範囲の中の、物体ではない部分を消すために使う)
struct Shape
{
	bool subtract = false;
	PolyPoints pts;
};
typedef std::vector<Shape> ShapeList;

//---------------------------------------------------------------------
//	ONNX Runtime の読み込み
//---------------------------------------------------------------------
static bool g_ort_ready = false;

// このビルドが使った onnxruntime のヘッダーの情報 (API版数とDirectML対応の有無)
static std::wstring build_info()
{
	std::wstring t = L"ヘッダーAPI " + std::to_wstring((int)ORT_API_VERSION);
#ifdef OM_HAS_DML
	t += L" / DML対応ビルド";
#else
	t += L" / DML非対応ビルド";
#endif
	return t;
}
static std::wstring g_ort_path;		// 実際に読み込んだ onnxruntime.dll のパス

static void init_ort()
{
	if (g_ort_ready)
	{
		return;
	}
	// 探す場所: onnxruntime_dml フォルダ (同じ階層 → 1つ上) → onnxruntime フォルダ → プラグインと同じフォルダ → 通常の検索順
	// (プラグインがPlugin\ObjectMask\などのサブフォルダにある場合は、その1つ上のonnxruntimeフォルダも探す)
	std::wstring dir = plugin_dir();
	std::vector<std::wstring> candidates;
	size_t up = dir.find_last_of(L"\\/");
	std::wstring parent = (up != std::wstring::npos) ? dir.substr(0, up) : std::wstring();
	candidates.push_back(dir + L"\\onnxruntime_dml\\onnxruntime.dll");	// このプラグイン専用 (DirectML版)。他のプラグインのONNX Runtimeと別バージョンにできる
	if (!parent.empty())
	{
		candidates.push_back(parent + L"\\onnxruntime_dml\\onnxruntime.dll");	// .aux2 がPluginのサブフォルダにある場合
	}
	candidates.push_back(dir + L"\\onnxruntime\\onnxruntime.dll");
	candidates.push_back(dir + L"\\onnxruntime.dll");
	if (!parent.empty())
	{
		candidates.push_back(parent + L"\\onnxruntime\\onnxruntime.dll");
	}
	// DirectML.dll を先に読み込んでおく。onnxruntime.dll はDLLを名前で探すので、読み込み済みのものが使われる
	// 探す場所: Plugin\DirectML\ → プラグインと同じフォルダ → Plugin\onnxruntime\ → 1つ上のフォルダの DirectML フォルダ
	{
		std::vector<std::wstring> dml_candidates;
		dml_candidates.push_back(dir + L"\\DirectML\\DirectML.dll");
		dml_candidates.push_back(dir + L"\\onnxruntime_dml\\DirectML.dll");
		dml_candidates.push_back(dir + L"\\DirectML.dll");
		dml_candidates.push_back(dir + L"\\onnxruntime\\DirectML.dll");
		if (up != std::wstring::npos)
		{
			dml_candidates.push_back(dir.substr(0, up) + L"\\DirectML\\DirectML.dll");
		}
		bool dml_loaded = false;
		for (const std::wstring& dml_path : dml_candidates)
		{
			if (GetFileAttributesW(dml_path.c_str()) == INVALID_FILE_ATTRIBUTES)
			{
				continue;
			}
			if (LoadLibraryExW(dml_path.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH))
			{
				log_info(L"AIオブジェクトマスク: DirectML.dll を使用します (" + dml_path + L")");
				dml_loaded = true;
				break;
			}
		}
		if (!dml_loaded)
		{
			log_info(L"AIオブジェクトマスク: " + dir + L"\\DirectML\\DirectML.dll が見つかりません (Windows標準のDirectMLを使うため、GPUで動かない場合があります)");
		}
	}

	HMODULE h = nullptr;
	for (const std::wstring& path : candidates)
	{
		if (GetFileAttributesW(path.c_str()) == INVALID_FILE_ATTRIBUTES)
		{
			continue;
		}
		// DirectML.dll など、同じフォルダにある依存DLLも一緒に見つけられるようにする
		h = LoadLibraryExW(path.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
		if (h)
		{
			break;
		}
	}
	if (!h)
	{
		// どこにも無い場合は通常の検索順で探す (Windows標準の古いDLLが見つかることがある)
		h = LoadLibraryW(L"onnxruntime.dll");
	}
	if (!h)
	{
		throw LoadError{ L"onnxruntime.dll を読み込めません。" + dir + L"\\onnxruntime\\ に置いてください" };
	}
	wchar_t loaded[MAX_PATH] = {};
	GetModuleFileNameW(h, loaded, MAX_PATH);

	typedef const OrtApiBase* (ORT_API_CALL* GetApiBaseFn)(void);
	GetApiBaseFn get_base = reinterpret_cast<GetApiBaseFn>(reinterpret_cast<void*>(GetProcAddress(h, "OrtGetApiBase")));
	if (!get_base)
	{
		throw LoadError{ std::wstring(L"onnxruntime.dll ではないファイルを読み込みました: ") + loaded };
	}
	const OrtApiBase* base = get_base();
	const OrtApi* api = base ? base->GetApi(ORT_API_VERSION) : nullptr;
	std::wstring version = (base && base->GetVersionString) ? widen(base->GetVersionString()) : L"不明";
	if (!api)
	{
		throw LoadError{ std::wstring(L"読み込まれた onnxruntime.dll (") + loaded + L", version " + version
			+ L") がビルドに使ったONNX Runtimeより古いため使えません。" + dir + L"\\onnxruntime\\ に、ビルドに使ったものと同じバージョンのonnxruntime.dllを置いてください" };
	}
	Ort::InitApi(api);
	g_ort_ready = true;
	g_ort_path = loaded;
	log_info(std::wstring(L"AIオブジェクトマスク: onnxruntime ") + version + L" を使用します (" + loaded + L")");
	log_info(L"AIオブジェクトマスク: このビルドのヘッダー: " + build_info());
}

// フィルタの現在の設定値
struct Params
{
	int bx = 0, by = 0, bw = 0, bh = 0;
	int anchor = 0;
	ShapeList shapes;				// 囲んだ範囲 (元画像座標。空なら枠だけを使う)
	bool track = true;
	int interval = 4;				// SAM2で検出し直す間隔 (フレーム)。間のフレームは物体の移動量だけを推定してマスクをずらす
	bool invert = false;
	float feather = 1.0f;
	float grow = 0.0f;
	float smooth = 0.3f;
	float threshold = 0.5f;			// 小さいほど、あいまいな部分も不透明(対象)として残す
	float refine = 6.0f;			// 輪郭補正の半径 (元画像の画素数。0で補正しない)
	float edge_color = 2.0f;		// 縁の色補正の幅 (元画像の画素数。0で補正しない)
	float matte = 2.0f;				// 縁の精密化の幅 (元画像の画素数。0で補正しない)
	float despeckle = 6.0f;			// ゴミ除去: これ以下の大きさ(画素数)の小さな塊や穴を消す (0で行わない)
	float alpha_low = 0.10f;		// 透明度の下限: これ以下の不透明度は、完全に透明にする
	float alpha_high = 0.90f;		// 透明度の上限: これ以上の不透明度は、完全に不透明にする
	float smooth_shape = 1.0f;		// 形のなめらか化: これより細い突起や細いくぼみをならす (元画像の画素数。0で行わない)
	float unify = 0.0f;				// 色の均一化: 物体の中の、周りと違う色の小さな点を、周りの色に塗りなおす範囲 (0で行わない)
};

// 1フレーム分のマスク
struct MaskEntry
{
	int w = 0, h = 0;
	std::vector<uint8_t> a;			// 0〜255 (255 = 残す)
	BoxF bbox;						// 元画像座標での物体の外接枠
	bool valid = false;				// 物体が見つかったか
	uint8_t sig[256] = {};			// フレーム内容の簡易シグネチャ (上流の変更を検出する)
	std::vector<uint8_t> thumb;		// 追跡用の縮小グレー画像
	std::vector<uint8_t> z;			// マスクの連長圧縮データ (a を手放した古いフレームはこれから復元する)
	int tw = 0, th = 0;
	int age = 0;					// 最後にSAM2で検出してからのフレーム数 (0 = SAM2で検出したフレーム)
	uint64_t chash = 0;				// フレームの内容のハッシュ (同じ内容のフレームでマスクを使い回す)
};

// エフェクト1個ごとの状態
// 同じパラメータ(範囲・基準フレーム・画像サイズなど)で共有するキャッシュ。
// エフェクトのインスタンスやプレビュー/出力の違いに関係なく共有し、ファイルにも保存する
struct CacheBucket
{
	std::map<int, MaskEntry> masks;
	std::vector<int> pending;		// まだファイルに書いていないフレーム
	std::unordered_map<uint64_t, int> by_hash;	// フレームの内容 → フレーム番号
	ULONGLONG last_used = 0;
	bool file_ready = false;		// ファイルのヘッダーが書き込み済みか
	int mw = 0, mh = 0;
	uint64_t key = 0;
};

//---------------------------------------------------------------------
//	SAM2 推論
//	対応モデル: エンコーダ(image -> image_embed, high_res_feats_0, high_res_feats_1)
//	            デコーダ(image_embed, high_res_feats_0/1, point_coords, point_labels,
//	                     mask_input, has_mask_input, orig_im_size -> masks, iou_predictions)
//	入出力名は読み込み時にログへ出力する
//---------------------------------------------------------------------
class Sam2
{
public:
	bool ensure_loaded(std::wstring& err)
	{
		if (m_loaded)
		{
			return true;
		}
		if (m_failed && m_fail_generation == g_generation.load())
		{
			err = m_err;
			return false;
		}
		m_failed = false;
		try
		{
			init_ort();
#ifndef OM_HAS_DML
			log_warn(L"AIオブジェクトマスク: このビルドはGPU(DirectML)に対応していません (dml_provider_factory.h が見つからなかったため)。CPUで実行します");
			m_cpu_reason = L"DirectML未対応のビルド";
#endif
			create_sessions_with_fallback();
			m_loaded = true;
			return true;
		}
		catch (const LoadError& e)
		{
			m_err = e.msg;
		}
		catch (const Ort::Exception& e)
		{
			m_err = L"ONNX Runtimeのエラー: " + widen(e.what());
		}
		catch (const std::exception& e)
		{
			m_err = L"読み込みエラー: " + widen(e.what());
		}
		m_failed = true;
		m_fail_generation = g_generation.load();
		m_loaded = false;
		err = m_err;
		log_error(L"AIオブジェクトマスク: " + m_err);
		return false;
	}

	// ps は元画像座標。out は mw*mh のマスク(0〜255)
	bool segment(const PIXEL_RGBA* img, int w, int h, const PromptSet& ps, int mw, int mh, std::vector<uint8_t>& out, std::wstring& err)
	{
		try
		{
			run(img, w, h, ps, mw, mh, out);
			return true;
		}
		catch (const Ort::Exception& e)
		{
			if (m_gpu)
			{
				// GPUでの実行に失敗した場合はCPUに切り替えてやり直す
				log_warn(L"AIオブジェクトマスク: GPU実行に失敗したためCPUに切り替えます: " + widen(e.what()));
				m_cpu_reason = L"GPU実行に失敗: " + short_text(widen(e.what()), 80);
				m_gpu = false;
				try
				{
					create_sessions();
					run(img, w, h, ps, mw, mh, out);
					return true;
				}
				catch (const Ort::Exception& e2)
				{
					err = L"ONNX Runtimeのエラー: " + widen(e2.what());
				}
				catch (const LoadError& e2)
				{
					err = e2.msg;
				}
			}
			else
			{
				err = L"ONNX Runtimeのエラー: " + widen(e.what());
			}
		}
		catch (const LoadError& e)
		{
			err = e.msg;
		}
		catch (const std::exception& e)
		{
			err = L"推論エラー: " + widen(e.what());
		}
		log_error(L"AIオブジェクトマスク: " + err);
		return false;
	}

	// 画面とログに出す、実行している機器の説明 (CPUのときは、GPUにならなかった理由と、読み込んだDLLのフォルダも出す)
	std::wstring device_name() const
	{
		if (m_gpu)
		{
			return L"GPU(DirectML)";
		}
		std::wstring text = L"CPU";
		if (!m_cpu_reason.empty())
		{
			text += L" - " + m_cpu_reason;
		}
		text += L" {" + build_info() + L"}";
		size_t p = g_ort_path.find_last_of(L"\\/");
		if (p != std::wstring::npos)
		{
			size_t q = g_ort_path.find_last_of(L"\\/", p - 1);
			text += L" [" + g_ort_path.substr(q == std::wstring::npos ? 0 : q + 1, p - (q == std::wstring::npos ? 0 : q + 1)) + L"]";
		}
		return text;
	}

private:
	std::unique_ptr<Ort::Env> m_env;
	std::unique_ptr<Ort::Session> m_enc;
	std::unique_ptr<Ort::Session> m_dec;
	std::vector<std::string> m_enc_in, m_enc_out, m_dec_in, m_dec_out;
	int m_size = 1024;			// エンコーダの入力解像度
	int m_mask_in_size = 256;	// デコーダのmask_inputの解像度
	bool m_loaded = false;
	bool m_failed = false;
	int m_fail_generation = -1;
	std::wstring m_err;
	std::wstring m_cpu_reason;		// CPUで動かしている理由
#ifdef OM_HAS_DML
	bool m_gpu = true;
#else
	bool m_gpu = false;
#endif
	std::vector<float> m_input;

	// GPU(DirectML)でモデルを作れない場合は、CPUに切り替えて作り直す
	void create_sessions_with_fallback()
	{
		try
		{
			create_sessions();
		}
		catch (const Ort::Exception& e)
		{
			if (!m_gpu)
			{
				throw;
			}
			log_warn(L"AIオブジェクトマスク: GPU(DirectML)を使えないためCPUで実行します: " + widen(e.what()));
			m_cpu_reason = short_text(widen(e.what()), 90);
			m_gpu = false;
			create_sessions();
		}
	}

	Ort::SessionOptions make_options()
	{
		Ort::SessionOptions so;
		so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
		so.SetIntraOpNumThreads(4);
#ifdef OM_HAS_DML
		if (m_gpu)
		{
			so.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
			so.DisableMemPattern();
			const OrtDmlApi* dml = nullptr;
			Ort::ThrowOnError(Ort::GetApi().GetExecutionProviderApi("DML", ORT_API_VERSION, reinterpret_cast<const void**>(&dml)));
			Ort::ThrowOnError(dml->SessionOptionsAppendExecutionProvider_DML(so, 0));
		}
#endif
		return so;
	}

	static void read_names(Ort::Session& s, std::vector<std::string>& in, std::vector<std::string>& out)
	{
		Ort::AllocatorWithDefaultOptions alloc;
		in.clear();
		out.clear();
		for (size_t i = 0; i < s.GetInputCount(); i++)
		{
			in.push_back(s.GetInputNameAllocated(i, alloc).get());
		}
		for (size_t i = 0; i < s.GetOutputCount(); i++)
		{
			out.push_back(s.GetOutputNameAllocated(i, alloc).get());
		}
	}

	static std::wstring join_names(const std::vector<std::string>& v)
	{
		std::wstring s;
		for (size_t i = 0; i < v.size(); i++)
		{
			if (i)
			{
				s += L", ";
			}
			s += widen(v[i].c_str());
		}
		return s;
	}

	void create_sessions()
	{
		std::wstring dir = plugin_dir() + L"\\models\\";
		std::wstring enc_path = dir + L"sam2_encoder.onnx";
		std::wstring dec_path = dir + L"sam2_decoder.onnx";
		if (GetFileAttributesW(enc_path.c_str()) == INVALID_FILE_ATTRIBUTES)
		{
			throw LoadError{ L"モデルが見つかりません: " + enc_path };
		}
		if (GetFileAttributesW(dec_path.c_str()) == INVALID_FILE_ATTRIBUTES)
		{
			throw LoadError{ L"モデルが見つかりません: " + dec_path };
		}

		if (!m_env)
		{
			m_env.reset(new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "ObjectMask"));
		}
		m_enc.reset();
		m_dec.reset();
		m_enc.reset(new Ort::Session(*m_env, enc_path.c_str(), make_options()));
		m_dec.reset(new Ort::Session(*m_env, dec_path.c_str(), make_options()));
		read_names(*m_enc, m_enc_in, m_enc_out);
		read_names(*m_dec, m_dec_in, m_dec_out);
		log_info(std::wstring(L"AIオブジェクトマスク: モデルを読み込みました (") + device_name() + L")");
		log_info(L"  encoder 入力: " + join_names(m_enc_in) + L" / 出力: " + join_names(m_enc_out));
		log_info(L"  decoder 入力: " + join_names(m_dec_in) + L" / 出力: " + join_names(m_dec_out));

		if (m_enc_in.empty())
		{
			throw LoadError{ L"エンコーダの入力がありません" };
		}
		// 入力解像度は静的な場合のみモデルの値を使う
		auto shape = m_enc->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
		if (shape.size() == 4 && shape[2] > 0 && shape[2] == shape[3])
		{
			m_size = (int)shape[2];
		}
		else
		{
			m_size = 1024;
		}
		m_mask_in_size = 256;
		for (size_t i = 0; i < m_dec_in.size(); i++)
		{
			if (m_dec_in[i] == "mask_input")
			{
				auto ms = m_dec->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
				if (ms.size() == 4 && ms[2] > 0 && ms[2] == ms[3])
				{
					m_mask_in_size = (int)ms[2];
				}
			}
		}
	}

	void run(const PIXEL_RGBA* img, int w, int h, const PromptSet& ps, int mw, int mh, std::vector<uint8_t>& out)
	{
		const int S = m_size;
		const size_t plane = (size_t)S * S;
		m_input.resize(plane * 3);

		// 前処理: S x S へ双線形リサイズして ImageNet の平均・分散で正規化 (NCHW)
		std::vector<int> xi0(S), xi1(S);
		std::vector<float> xw(S);
		for (int x = 0; x < S; x++)
		{
			float fx = (x + 0.5f) * w / S - 0.5f;
			int i0 = (int)std::floor(fx);
			xw[x] = fx - i0;
			xi0[x] = std::clamp(i0, 0, w - 1);
			xi1[x] = std::clamp(i0 + 1, 0, w - 1);
		}
		const float mean[3] = { 0.485f, 0.456f, 0.406f };
		const float istd[3] = { 1.0f / 0.229f, 1.0f / 0.224f, 1.0f / 0.225f };
		parallel_for(S, [&](int y_begin, int y_end)
			{
				for (int y = y_begin; y < y_end; y++)
				{
					float fy = (y + 0.5f) * h / S - 0.5f;
					int j0 = (int)std::floor(fy);
					float wy = fy - j0;
					const PIXEL_RGBA* r0 = img + (size_t)std::clamp(j0, 0, h - 1) * w;
					const PIXEL_RGBA* r1 = img + (size_t)std::clamp(j0 + 1, 0, h - 1) * w;
					for (int x = 0; x < S; x++)
					{
						const PIXEL_RGBA& p00 = r0[xi0[x]];
						const PIXEL_RGBA& p01 = r0[xi1[x]];
						const PIXEL_RGBA& p10 = r1[xi0[x]];
						const PIXEL_RGBA& p11 = r1[xi1[x]];
						float wx = xw[x];
						float r = (p00.r * (1 - wx) + p01.r * wx) * (1 - wy) + (p10.r * (1 - wx) + p11.r * wx) * wy;
						float g = (p00.g * (1 - wx) + p01.g * wx) * (1 - wy) + (p10.g * (1 - wx) + p11.g * wx) * wy;
						float b = (p00.b * (1 - wx) + p01.b * wx) * (1 - wy) + (p10.b * (1 - wx) + p11.b * wx) * wy;
						size_t o = (size_t)y * S + x;
						m_input[o] = (r / 255.0f - mean[0]) * istd[0];
						m_input[plane + o] = (g / 255.0f - mean[1]) * istd[1];
						m_input[plane * 2 + o] = (b / 255.0f - mean[2]) * istd[2];
					}
				}
			});

		Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

		// エンコーダ
		int64_t in_shape[4] = { 1, 3, S, S };
		Ort::Value in_tensor = Ort::Value::CreateTensor<float>(mem, m_input.data(), m_input.size(), in_shape, 4);
		const char* enc_in_names[1] = { m_enc_in[0].c_str() };
		std::vector<const char*> enc_out_names;
		for (auto& n : m_enc_out)
		{
			enc_out_names.push_back(n.c_str());
		}
		std::vector<Ort::Value> eo = m_enc->Run(Ort::RunOptions{ nullptr }, enc_in_names, &in_tensor, 1, enc_out_names.data(), enc_out_names.size());

		// デコーダの入力を組み立てる (枠: 左上=2, 右下=3 の後に、物体上の点=1 / 背景上の点=0 を続ける)
		const size_t n_pts = 2 + ps.labels.size();
		std::vector<float> coords(n_pts * 2);
		std::vector<float> labels_f(n_pts);
		coords[0] = ps.box.x0 * S / w;
		coords[1] = ps.box.y0 * S / h;
		coords[2] = ps.box.x1 * S / w;
		coords[3] = ps.box.y1 * S / h;
		labels_f[0] = 2.0f;
		labels_f[1] = 3.0f;
		for (size_t i = 0; i < ps.labels.size(); i++)
		{
			coords[(2 + i) * 2] = ps.pts[i * 2] * S / w;
			coords[(2 + i) * 2 + 1] = ps.pts[i * 2 + 1] * S / h;
			labels_f[2 + i] = ps.labels[i];
		}
		std::vector<int32_t> labels_i32(labels_f.begin(), labels_f.end());
		std::vector<int64_t> labels_i64(labels_f.begin(), labels_f.end());
		std::vector<float> mask_in((size_t)m_mask_in_size * m_mask_in_size, 0.0f);
		float has_mask[1] = { 0.0f };
		int64_t orig_size[2] = { mh, mw };
		int64_t coord_shape[3] = { 1, (int64_t)n_pts, 2 };
		int64_t label_shape[2] = { 1, (int64_t)n_pts };
		int64_t mask_shape[4] = { 1, 1, m_mask_in_size, m_mask_in_size };
		int64_t has_shape[1] = { 1 };
		int64_t orig_shape[1] = { 2 };

		std::vector<Ort::Value> dec_values;
		std::vector<const char*> dec_names;
		int32_t orig_size_i32[2] = { mh, mw };
		for (size_t di = 0; di < m_dec_in.size(); di++)
		{
			const std::string& name = m_dec_in[di];
			dec_names.push_back(name.c_str());
			// 入力の型はモデルによって異なるため、モデル側の型に合わせる
			auto input_type = m_dec->GetInputTypeInfo(di).GetTensorTypeAndShapeInfo().GetElementType();
			if (name == "point_coords")
			{
				dec_values.push_back(Ort::Value::CreateTensor<float>(mem, coords.data(), coords.size(), coord_shape, 3));
			}
			else if (name == "point_labels")
			{
				if (input_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32)
				{
					dec_values.push_back(Ort::Value::CreateTensor<int32_t>(mem, labels_i32.data(), labels_i32.size(), label_shape, 2));
				}
				else if (input_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)
				{
					dec_values.push_back(Ort::Value::CreateTensor<int64_t>(mem, labels_i64.data(), labels_i64.size(), label_shape, 2));
				}
				else
				{
					dec_values.push_back(Ort::Value::CreateTensor<float>(mem, labels_f.data(), labels_f.size(), label_shape, 2));
				}
			}
			else if (name == "mask_input")
			{
				dec_values.push_back(Ort::Value::CreateTensor<float>(mem, mask_in.data(), mask_in.size(), mask_shape, 4));
			}
			else if (name == "has_mask_input")
			{
				dec_values.push_back(Ort::Value::CreateTensor<float>(mem, has_mask, 1, has_shape, 1));
			}
			else if (name == "orig_im_size")
			{
				if (input_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32)
				{
					dec_values.push_back(Ort::Value::CreateTensor<int32_t>(mem, orig_size_i32, 2, orig_shape, 1));
				}
				else
				{
					dec_values.push_back(Ort::Value::CreateTensor<int64_t>(mem, orig_size, 2, orig_shape, 1));
				}
			}
			else
			{
				// エンコーダの出力と同名の入力はそのまま渡す
				bool found = false;
				for (size_t k = 0; k < m_enc_out.size(); k++)
				{
					if (m_enc_out[k] == name)
					{
						dec_values.push_back(std::move(eo[k]));
						found = true;
						break;
					}
				}
				if (!found)
				{
					throw LoadError{ L"デコーダの入力名に対応していません: " + widen(name.c_str()) };
				}
			}
		}

		int mask_idx = 0;
		int iou_idx = -1;
		std::vector<const char*> dec_out_names;
		for (size_t i = 0; i < m_dec_out.size(); i++)
		{
			dec_out_names.push_back(m_dec_out[i].c_str());
			if (m_dec_out[i] == "masks")
			{
				mask_idx = (int)i;
			}
			if (m_dec_out[i] == "iou_predictions")
			{
				iou_idx = (int)i;
			}
		}
		std::vector<Ort::Value> dout = m_dec->Run(Ort::RunOptions{ nullptr }, dec_names.data(), dec_values.data(), dec_values.size(), dec_out_names.data(), dec_out_names.size());

		// 出力から最もスコアの高いマスクを選ぶ
		auto info = dout[mask_idx].GetTensorTypeAndShapeInfo();
		auto shape = info.GetShape();
		size_t nd = shape.size();
		if (nd < 2)
		{
			throw LoadError{ L"デコーダのマスク出力の形状が想定外です" };
		}
		int64_t H = shape[nd - 2];
		int64_t W = shape[nd - 1];
		int64_t N = (nd >= 4) ? shape[nd - 3] : 1;
		if (H <= 0 || W <= 0 || N <= 0)
		{
			throw LoadError{ L"デコーダのマスク出力の形状が想定外です" };
		}
		int best = 0;
		if (iou_idx >= 0 && N > 1)
		{
			auto iou_info = dout[iou_idx].GetTensorTypeAndShapeInfo();
			if (iou_info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
			{
				const float* iou = dout[iou_idx].GetTensorData<float>();
				size_t count = iou_info.GetElementCount();
				float bv = -1e30f;
				for (size_t i = 0; i < count && i < (size_t)N; i++)
				{
					if (iou[i] > bv)
					{
						bv = iou[i];
						best = (int)i;
					}
				}
			}
		}

		std::vector<float> plane_f((size_t)H * W);
		auto etype = info.GetElementType();
		size_t offset = (size_t)best * H * W;
		if (etype == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
		{
			const float* p = dout[mask_idx].GetTensorData<float>() + offset;
			std::copy(p, p + (size_t)H * W, plane_f.begin());
		}
		else if (etype == ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL || etype == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8)
		{
			const uint8_t* p = dout[mask_idx].GetTensorData<uint8_t>() + offset;
			for (size_t i = 0; i < (size_t)H * W; i++)
			{
				plane_f[i] = p[i] ? 1.0f : 0.0f;
			}
		}
		else
		{
			throw LoadError{ L"デコーダのマスク出力の型に対応していません" };
		}

		// ロジットなら sigmoid で 0〜1 にする (すでに 0〜1 ならそのまま)
		float minv = 1e30f, maxv = -1e30f;
		for (float v : plane_f)
		{
			minv = std::min(minv, v);
			maxv = std::max(maxv, v);
		}
		bool logits = (minv < 0.0f || maxv > 1.0f);

		out.assign((size_t)mw * mh, 0);
		for (int y = 0; y < mh; y++)
		{
			float fy = (y + 0.5f) * H / mh - 0.5f;
			int j0 = (int)std::floor(fy);
			float wy = fy - j0;
			int ja = (int)std::clamp<int64_t>(j0, 0, H - 1);
			int jb = (int)std::clamp<int64_t>(j0 + 1, 0, H - 1);
			for (int x = 0; x < mw; x++)
			{
				float fx = (x + 0.5f) * W / mw - 0.5f;
				int i0 = (int)std::floor(fx);
				float wx = fx - i0;
				int ia = (int)std::clamp<int64_t>(i0, 0, W - 1);
				int ib = (int)std::clamp<int64_t>(i0 + 1, 0, W - 1);
				float v = (plane_f[(size_t)ja * W + ia] * (1 - wx) + plane_f[(size_t)ja * W + ib] * wx) * (1 - wy)
					+ (plane_f[(size_t)jb * W + ia] * (1 - wx) + plane_f[(size_t)jb * W + ib] * wx) * wy;
				if (logits)
				{
					// sigmoidだと確信の弱い部分が半透明になるため、傾きを急にして、ほぼ0か1にする
					v = std::clamp(0.5f + v * 0.4f, 0.0f, 1.0f);
				}
				out[(size_t)y * mw + x] = (uint8_t)std::clamp((int)std::lround(v * 255.0f), 0, 255);
			}
		}
	}
};

static Sam2* g_sam = nullptr;

static Sam2& sam()
{
	if (!g_sam)
	{
		g_sam = new Sam2();
	}
	return *g_sam;
}

//---------------------------------------------------------------------
//	プレビュー共有 (フィルタ → 枠選択ウィンドウ)
//---------------------------------------------------------------------
struct PreviewState
{
	std::mutex mtx;
	std::vector<uint32_t> bgra;
	int pw = 0, ph = 0;
	int src_w = 0, src_h = 0;
	int frame = 0;
	uint64_t version = 0;
	int bx = 0, by = 0, bw = 0, bh = 0;
	bool has_bbox = false;
	BoxF bbox;
	ShapeList shapes;
	int anchor = 0;
	std::wstring status;
};

static PreviewState g_preview;

// 直前の処理結果 (静止画のように同じ入力が続くときは、これをそのまま返す)
struct LastResult
{
	uint64_t key = 0;
	int w = 0, h = 0;
	std::vector<PIXEL_RGBA> img;	// マスク適用後の画像
	std::vector<float> m;			// プレビュー用のマスク
	int mw = 0, mh = 0;
	BoxF bbox;
	bool has_bbox = false;
	std::wstring msg;
	bool valid = false;
};
static std::mutex g_result_mtx;
static LastResult g_last_result;
static std::atomic<uint64_t> g_prev_chash{ 0 };
static std::atomic<bool> g_preview_visible{ true };		// 枠選択ウィンドウが表示されているか
static std::atomic<ULONGLONG> g_last_call{ 0 };			// 最後にフィルタが呼ばれた時刻
static std::atomic<ULONGLONG> g_last_publish{ 0 };		// 最後にプレビューを渡した時刻

static void set_status(const std::wstring& status)
{
	std::lock_guard<std::mutex> lock(g_preview.mtx);
	if (g_preview.status != status)
	{
		g_preview.status = status;
		g_preview.version++;
	}
}

// mask が null でなければ、対象外を暗くして重ねる
static void publish_preview(const PIXEL_RGBA* img, int w, int h, int frame, const Params& p,
	const std::vector<float>* mask, int mw, int mh, const BoxF* bbox, const std::wstring& status)
{
	int pw = std::min(w, PREVIEW_MAX_WIDTH);
	int ph = std::max(1, (int)((int64_t)h * pw / w));
	std::vector<uint32_t> px((size_t)pw * ph);
	for (int y = 0; y < ph; y++)
	{
		int sy = std::min(h - 1, (int)((int64_t)y * h / ph));
		int my = mask ? std::min(mh - 1, (int)((int64_t)y * mh / ph)) : 0;
		for (int x = 0; x < pw; x++)
		{
			int sx = std::min(w - 1, (int)((int64_t)x * w / pw));
			const PIXEL_RGBA& s = img[(size_t)sy * w + sx];
			float shade = 1.0f;
			if (mask)
			{
				int mx = std::min(mw - 1, (int)((int64_t)x * mw / pw));
				float v = (*mask)[(size_t)my * mw + mx];
				if (p.invert)
				{
					v = 1.0f - v;
				}
				shade = 0.3f + 0.7f * v;
			}
			uint32_t r = (uint32_t)(s.r * shade);
			uint32_t g = (uint32_t)(s.g * shade);
			uint32_t b = (uint32_t)(s.b * shade);
			px[(size_t)y * pw + x] = 0xff000000u | (r << 16) | (g << 8) | b;
		}
	}
	std::lock_guard<std::mutex> lock(g_preview.mtx);
	g_preview.bgra.swap(px);
	g_preview.pw = pw;
	g_preview.ph = ph;
	g_preview.src_w = w;
	g_preview.src_h = h;
	g_preview.frame = frame;
	g_preview.bx = p.bx;
	g_preview.by = p.by;
	g_preview.bw = p.bw;
	g_preview.bh = p.bh;
	g_preview.shapes = p.shapes;
	g_preview.anchor = p.anchor;
	g_preview.has_bbox = (bbox != nullptr);
	if (bbox)
	{
		g_preview.bbox = *bbox;
	}
	g_preview.status = status;
	g_preview.version++;
}

//---------------------------------------------------------------------
//	マスクの後処理
//---------------------------------------------------------------------
static void box_blur_h(const float* src, float* dst, int w, int h, int r)
{
	const float inv = 1.0f / (2 * r + 1);
	for (int y = 0; y < h; y++)
	{
		const float* s = src + (size_t)y * w;
		float* d = dst + (size_t)y * w;
		float sum = 0.0f;
		for (int i = -r; i <= r; i++)
		{
			sum += s[std::clamp(i, 0, w - 1)];
		}
		for (int x = 0; x < w; x++)
		{
			d[x] = sum * inv;
			sum += s[std::min(x + r + 1, w - 1)] - s[std::max(x - r, 0)];
		}
	}
}

static void box_blur_v(const float* src, float* dst, int w, int h, int r)
{
	const float inv = 1.0f / (2 * r + 1);
	for (int x = 0; x < w; x++)
	{
		float sum = 0.0f;
		for (int i = -r; i <= r; i++)
		{
			sum += src[(size_t)std::clamp(i, 0, h - 1) * w + x];
		}
		for (int y = 0; y < h; y++)
		{
			dst[(size_t)y * w + x] = sum * inv;
			sum += src[(size_t)std::min(y + r + 1, h - 1) * w + x] - src[(size_t)std::max(y - r, 0) * w + x];
		}
	}
}

static void blur(std::vector<float>& m, int w, int h, int r)
{
	std::vector<float> tmp(m.size());
	for (int pass = 0; pass < 2; pass++)
	{
		box_blur_h(m.data(), tmp.data(), w, h, r);
		box_blur_v(tmp.data(), m.data(), w, h, r);
	}
}

static void morph(std::vector<float>& m, int w, int h, int r, bool dilate)
{
	std::vector<float> tmp(m.size());
	for (int y = 0; y < h; y++)
	{
		for (int x = 0; x < w; x++)
		{
			float v = m[(size_t)y * w + x];
			for (int k = -r; k <= r; k++)
			{
				float t = m[(size_t)y * w + std::clamp(x + k, 0, w - 1)];
				v = dilate ? std::max(v, t) : std::min(v, t);
			}
			tmp[(size_t)y * w + x] = v;
		}
	}
	for (int y = 0; y < h; y++)
	{
		for (int x = 0; x < w; x++)
		{
			float v = tmp[(size_t)y * w + x];
			for (int k = -r; k <= r; k++)
			{
				float t = tmp[(size_t)std::clamp(y + k, 0, h - 1) * w + x];
				v = dilate ? std::max(v, t) : std::min(v, t);
			}
			m[(size_t)y * w + x] = v;
		}
	}
}

// 小さなゴミを消す: 面積が area_thr 以下の「物体の小さな塊」を消し、「物体に囲まれた小さな穴」を埋める (マスクの解像度で処理)
static void remove_specks(std::vector<float>& m, int w, int h, int area_thr)
{
	if (area_thr < 1)
	{
		return;
	}
	const size_t n = (size_t)w * h;
	std::vector<uint8_t> visited(n, 0);
	std::vector<size_t> stack;
	std::vector<size_t> comp;
	for (size_t start = 0; start < n; start++)
	{
		if (visited[start])
		{
			continue;
		}
		const bool fg = m[start] >= 0.5f;
		bool touches_border = false;
		comp.clear();
		stack.clear();
		stack.push_back(start);
		visited[start] = 1;
		while (!stack.empty())
		{
			size_t i = stack.back();
			stack.pop_back();
			comp.push_back(i);
			int x = (int)(i % w);
			int y = (int)(i / w);
			if (x == 0 || y == 0 || x == w - 1 || y == h - 1)
			{
				touches_border = true;
			}
			const int nx[4] = { x - 1, x + 1, x, x };
			const int ny[4] = { y, y, y - 1, y + 1 };
			for (int k = 0; k < 4; k++)
			{
				if (nx[k] < 0 || nx[k] >= w || ny[k] < 0 || ny[k] >= h)
				{
					continue;
				}
				size_t j = (size_t)ny[k] * w + nx[k];
				if (!visited[j] && ((m[j] >= 0.5f) == fg))
				{
					visited[j] = 1;
					stack.push_back(j);
				}
			}
		}
		if ((int)comp.size() > area_thr)
		{
			continue;
		}
		if (fg)
		{
			for (size_t i : comp)
			{
				m[i] = 0.0f;
			}
		}
		else if (!touches_border)
		{
			for (size_t i : comp)
			{
				m[i] = 1.0f;
			}
		}
	}
}

// cur と前フレームのマスクから、拡縮・ぼかし後の 0〜1 のマスクを作る (scale = マスク解像度 / 元画像解像度)
static void build_float_mask(const MaskEntry& cur, const MaskEntry* prev, const Params& p, float scale, std::vector<float>& m, bool with_grow)
{
	const int w = cur.w;
	const int h = cur.h;
	m.resize((size_t)w * h);
	for (size_t i = 0; i < m.size(); i++)
	{
		m[i] = cur.a[i] / 255.0f;
	}
	if (prev && prev->valid && prev->w == w && prev->h == h && p.smooth > 0.0f)
	{
		float s = std::clamp(p.smooth, 0.0f, 0.95f);
		for (size_t i = 0; i < m.size(); i++)
		{
			m[i] = m[i] * (1.0f - s) + (prev->a[i] / 255.0f) * s;
		}
	}
	// 薄く半透明にならないよう、しきい値を境に不透明/透明をはっきりさせる (縁は後の拡大とぼかしで滑らかになる)
	{
		float th = std::clamp(p.threshold, 0.05f, 0.95f);
		for (float& v : m)
		{
			v = std::clamp((v - th) * 8.0f + 0.5f, 0.0f, 1.0f);
		}
	}
	// 拡大縮小は、出力用では元の画像の大きさで連続的に行う (ここでは、プレビュー用の大まかな処理だけ)
	if (p.despeckle >= 1.0f)
	{
		float d = p.despeckle * scale;
		remove_specks(m, w, h, std::max(1, (int)std::lround(0.785f * d * d)));
	}
	int gr = with_grow ? (int)std::lround(std::fabs(p.grow) * scale) : 0;
	if (gr > 0)
	{
		morph(m, w, h, gr, p.grow > 0.0f);
	}
	int br = (int)std::lround(p.feather * scale);
	if (br > 0)
	{
		blur(m, w, h, br);
	}
}

//---------------------------------------------------------------------
//	輪郭補正 (元の画像の輪郭に沿うように、マスクの縁を整える)
//	粗いマスクを元の大きさへ拡大すると縁が階段状になるので、画像の色の境目(輝度の差)に
//	吸い付くように補正する。ガイデッドフィルタを、処理を軽くするため縮小して計算する
//---------------------------------------------------------------------
// マスク(mw*mh)を元の大きさ(w*h)へ双線形で拡大する
static void upsample_mask(const std::vector<float>& m, int mw, int mh, int w, int h, std::vector<float>& out)
{
	out.resize((size_t)w * h);
	std::vector<int> xa(w), xb(w);
	std::vector<float> xw(w);
	for (int x = 0; x < w; x++)
	{
		float fx = (x + 0.5f) * mw / w - 0.5f;
		int i0 = (int)std::floor(fx);
		xw[x] = fx - i0;
		xa[x] = std::clamp(i0, 0, mw - 1);
		xb[x] = std::clamp(i0 + 1, 0, mw - 1);
	}
	parallel_for(h, [&](int y_begin, int y_end)
		{
			for (int y = y_begin; y < y_end; y++)
			{
				float fy = (y + 0.5f) * mh / h - 0.5f;
				int j0 = (int)std::floor(fy);
				float wy = fy - j0;
				const float* r0 = m.data() + (size_t)std::clamp(j0, 0, mh - 1) * mw;
				const float* r1 = m.data() + (size_t)std::clamp(j0 + 1, 0, mh - 1) * mw;
				float* row = out.data() + (size_t)y * w;
				for (int x = 0; x < w; x++)
				{
					row[x] = (r0[xa[x]] * (1 - xw[x]) + r0[xb[x]] * xw[x]) * (1 - wy)
						+ (r1[xa[x]] * (1 - xw[x]) + r1[xb[x]] * xw[x]) * wy;
				}
			}
		});
}

// 半径 r の箱形の平均 (積分画像。端は範囲内の画素だけで平均する)
static void box_mean(const std::vector<float>& src, int w, int h, int r, std::vector<float>& dst)
{
	std::vector<double> integral((size_t)(w + 1) * (h + 1), 0.0);
	for (int y = 0; y < h; y++)
	{
		double row = 0.0;
		for (int x = 0; x < w; x++)
		{
			row += src[(size_t)y * w + x];
			integral[(size_t)(y + 1) * (w + 1) + (x + 1)] = integral[(size_t)y * (w + 1) + (x + 1)] + row;
		}
	}
	dst.resize((size_t)w * h);
	for (int y = 0; y < h; y++)
	{
		int y0 = std::max(0, y - r);
		int y1 = std::min(h - 1, y + r);
		for (int x = 0; x < w; x++)
		{
			int x0 = std::max(0, x - r);
			int x1 = std::min(w - 1, x + r);
			double sum = integral[(size_t)(y1 + 1) * (w + 1) + (x1 + 1)] - integral[(size_t)y0 * (w + 1) + (x1 + 1)]
				- integral[(size_t)(y1 + 1) * (w + 1) + x0] + integral[(size_t)y0 * (w + 1) + x0];
			dst[(size_t)y * w + x] = (float)(sum / ((double)(x1 - x0 + 1) * (y1 - y0 + 1)));
		}
	}
}

// 縮小 (s x s の平均)
static void downscale_avg(const std::vector<float>& src, int w, int h, int s, int lw, int lh, std::vector<float>& dst)
{
	dst.assign((size_t)lw * lh, 0.0f);
	for (int y = 0; y < lh; y++)
	{
		for (int x = 0; x < lw; x++)
		{
			float sum = 0.0f;
			int n = 0;
			for (int dy = 0; dy < s; dy++)
			{
				int yy = y * s + dy;
				if (yy >= h)
				{
					break;
				}
				for (int dx = 0; dx < s; dx++)
				{
					int xx = x * s + dx;
					if (xx >= w)
					{
						break;
					}
					sum += src[(size_t)yy * w + xx];
					n++;
				}
			}
			dst[(size_t)y * lw + x] = n ? sum / n : 0.0f;
		}
	}
}

// m (mw*mh) を元の大きさへ拡大し、画像の輪郭に沿うように補正した不透明度 q (w*h) を作る
static void refine_alpha(const PIXEL_RGBA* img, int w, int h, const std::vector<float>& m, int mw, int mh,
	float radius, float gain, std::vector<float>& q)
{
	std::vector<float> p;
	upsample_mask(m, mw, mh, w, h, p);

	// ガイド画像 (輝度 0〜1)
	std::vector<float> guide((size_t)w * h);
	parallel_for(h, [&](int y_begin, int y_end)
		{
			for (int y = y_begin; y < y_end; y++)
			{
				const PIXEL_RGBA* row = img + (size_t)y * w;
				float* g = guide.data() + (size_t)y * w;
				for (int x = 0; x < w; x++)
				{
					g[x] = (row[x].r * 77 + row[x].g * 150 + row[x].b * 29) / (256.0f * 255.0f);
				}
			}
		});

	const int s = (std::min(w, h) >= 400) ? 2 : 1;
	const int lw = (w + s - 1) / s;
	const int lh = (h + s - 1) / s;
	const int rl = std::max(1, (int)std::lround(radius / s));
	const float eps = 0.002f;

	std::vector<float> gl, pl;
	downscale_avg(guide, w, h, s, lw, lh, gl);
	downscale_avg(p, w, h, s, lw, lh, pl);

	std::vector<float> mean_g, mean_p, gg(gl.size()), gp(gl.size()), mean_gg, mean_gp;
	for (size_t i = 0; i < gl.size(); i++)
	{
		gg[i] = gl[i] * gl[i];
		gp[i] = gl[i] * pl[i];
	}
	box_mean(gl, lw, lh, rl, mean_g);
	box_mean(pl, lw, lh, rl, mean_p);
	box_mean(gg, lw, lh, rl, mean_gg);
	box_mean(gp, lw, lh, rl, mean_gp);
	std::vector<float> a(gl.size()), b(gl.size());
	for (size_t i = 0; i < gl.size(); i++)
	{
		float var = mean_gg[i] - mean_g[i] * mean_g[i];
		float cov = mean_gp[i] - mean_g[i] * mean_p[i];
		a[i] = cov / (var + eps);
		b[i] = mean_p[i] - a[i] * mean_g[i];
	}
	std::vector<float> mean_a, mean_b;
	box_mean(a, lw, lh, rl, mean_a);
	box_mean(b, lw, lh, rl, mean_b);

	// 元の大きさへ戻して適用する
	q.resize((size_t)w * h);
	parallel_for(h, [&](int y_begin, int y_end)
		{
			for (int y = y_begin; y < y_end; y++)
			{
				float fy = (y + 0.5f) / s - 0.5f;
				int j0 = (int)std::floor(fy);
				float wy = fy - j0;
				int ja = std::clamp(j0, 0, lh - 1);
				int jb = std::clamp(j0 + 1, 0, lh - 1);
				for (int x = 0; x < w; x++)
				{
					float fx = (x + 0.5f) / s - 0.5f;
					int i0 = (int)std::floor(fx);
					float wx = fx - i0;
					int ia = std::clamp(i0, 0, lw - 1);
					int ib = std::clamp(i0 + 1, 0, lw - 1);
					auto bil = [&](const std::vector<float>& v)
						{
							return (v[(size_t)ja * lw + ia] * (1 - wx) + v[(size_t)ja * lw + ib] * wx) * (1 - wy)
								+ (v[(size_t)jb * lw + ia] * (1 - wx) + v[(size_t)jb * lw + ib] * wx) * wy;
						};
					float v = bil(mean_a) * guide[(size_t)y * w + x] + bil(mean_b);
					q[(size_t)y * w + x] = std::clamp((v - 0.5f) * gain + 0.5f, 0.0f, 1.0f);
				}
			}
		});
}

// 箱形ブラー (元の画像の大きさ用。縁は端の画素を延ばして計算する)
static void box_blur_full(std::vector<float>& img, int w, int h, int r)
{
	std::vector<float> tmp((size_t)w * h);
	const float inv = 1.0f / (2 * r + 1);
	parallel_for(h, [&](int y_begin, int y_end)
		{
			for (int y = y_begin; y < y_end; y++)
			{
				const float* src = img.data() + (size_t)y * w;
				float* dst = tmp.data() + (size_t)y * w;
				float sum = 0.0f;
				for (int i = -r; i <= r; i++)
				{
					sum += src[std::clamp(i, 0, w - 1)];
				}
				for (int x = 0; x < w; x++)
				{
					dst[x] = sum * inv;
					sum += src[std::min(x + r + 1, w - 1)] - src[std::max(x - r, 0)];
				}
			}
		});
	parallel_for(w, [&](int x_begin, int x_end)
		{
			for (int x = x_begin; x < x_end; x++)
			{
				float sum = 0.0f;
				for (int i = -r; i <= r; i++)
				{
					sum += tmp[(size_t)std::clamp(i, 0, h - 1) * w + x];
				}
				for (int y = 0; y < h; y++)
				{
					img[(size_t)y * w + x] = sum * inv;
					sum += tmp[(size_t)std::min(y + r + 1, h - 1) * w + x] - tmp[(size_t)std::max(y - r, 0) * w + x];
				}
			}
		});
}

// 不透明度 q の縁を、grow 画素ぶん外側(+)/内側(-)へ動かす。小数の値でも連続的に効く
static void grow_alpha(std::vector<float>& q, int w, int h, float grow)
{
	float g = std::fabs(grow);
	if (g < 0.05f)
	{
		return;
	}
	// ブラーして、しきい値を動かして、縁をはっきりさせ直す
	int r = std::max(1, (int)std::lround(g));
	float width = (float)(2 * r + 1);
	std::vector<float> b(q);
	box_blur_full(b, w, h, r);
	float thr = (grow > 0.0f) ? 0.5f - g / width : 0.5f + g / width;
	// しきい値の前後 delta の範囲で 0 から 1 へ切り替える。範囲が 0〜1 からはみ出さないようにする
	// (はみ出すと、物体から離れた場所の不透明度が 0 や 1 にならず、薄く残ってしまう)
	float gain = std::clamp(width / 1.5f, 2.0f, 10.0f);
	float delta = std::min({ 0.5f / gain, thr, 1.0f - thr });
	float lo = thr - delta;
	float inv_range = 1.0f / (2.0f * delta);
	parallel_for(h, [&](int y_begin, int y_end)
		{
			for (int y = y_begin; y < y_end; y++)
			{
				float* row = q.data() + (size_t)y * w;
				const float* br = b.data() + (size_t)y * w;
				for (int x = 0; x < w; x++)
				{
					row[x] = std::clamp((br[x] - lo) * inv_range, 0.0f, 1.0f);
				}
			}
		});
}

// 縁の色補正: 物体の縁の画素には、背景の色が混ざっている (にじみ)。
// 縁から depth 画素以内の画素の色を、その内側にある画素の色で置き換える。vis は見える部分の不透明度 (0〜1)
static void decontaminate_edges(PIXEL_RGBA* buf, const std::vector<float>& vis, int w, int h, float depth)
{
	if (depth < 0.5f)
	{
		return;
	}
	// 縁から depth 画素より内側の部分 (確実に物体の色の画素)
	std::vector<float> interior(vis);
	grow_alpha(interior, w, h, -depth);
	const size_t n = (size_t)w * h;
	std::vector<float> cr(n), cg(n), cb(n);
	parallel_for(h, [&](int y_begin, int y_end)
		{
			for (int y = y_begin; y < y_end; y++)
			{
				for (int x = 0; x < w; x++)
				{
					size_t i = (size_t)y * w + x;
					float wi = interior[i];
					cr[i] = buf[i].r * wi;
					cg[i] = buf[i].g * wi;
					cb[i] = buf[i].b * wi;
				}
			}
		});
	const int r = std::max(2, (int)std::ceil(depth * 2.0f));
	std::vector<float> wsum(interior);
	box_blur_full(cr, w, h, r);
	box_blur_full(cg, w, h, r);
	box_blur_full(cb, w, h, r);
	box_blur_full(wsum, w, h, r);
	parallel_for(h, [&](int y_begin, int y_end)
		{
			for (int y = y_begin; y < y_end; y++)
			{
				for (int x = 0; x < w; x++)
				{
					size_t i = (size_t)y * w + x;
					if (vis[i] <= 0.001f || wsum[i] < 0.02f)
					{
						continue;
					}
					// 内側の画素ほど元の色を残し、縁に近いほど置き換える
					float k = std::clamp(1.0f - interior[i], 0.0f, 1.0f);
					float inv = 1.0f / wsum[i];
					float nr = cr[i] * inv;
					float ng = cg[i] * inv;
					float nb = cb[i] * inv;
					buf[i].r = (unsigned char)std::clamp((int)std::lround(buf[i].r * (1 - k) + nr * k), 0, 255);
					buf[i].g = (unsigned char)std::clamp((int)std::lround(buf[i].g * (1 - k) + ng * k), 0, 255);
					buf[i].b = (unsigned char)std::clamp((int)std::lround(buf[i].b * (1 - k) + nb * k), 0, 255);
				}
			}
		});
}

// 縁の精密化: 縁の近く (縁から band 画素以内) の画素ごとに、「物体の色」と「背景の色」の混ざり具合を、
// 実際の画像の色から求め直す (アルファマット)。縁の画素の色 C は、近くの物体の色 F と背景の色 B の混合なので、
// 混ざり具合 = (C-B)・(F-B) / |F-B|^2 になる。これで、SAM2のマスクが多少ずれていても、縁が画像の輪郭に沿う。
// 物体と背景の色が近くて求められない場所は、元のマスクのままにする。recolor が true なら、縁の色を物体の色にする
static void matte_refine(PIXEL_RGBA* buf, std::vector<float>& q, int w, int h, float band, bool recolor)
{
	if (band < 0.5f)
	{
		return;
	}
	const size_t n = (size_t)w * h;
	// I: 確実に物体の部分、E: 確実に背景の部分 (どちらでもない部分が、縁の帯)
	std::vector<float> I(q), E(q);
	grow_alpha(I, w, h, -band);
	grow_alpha(E, w, h, band);
	for (size_t i = 0; i < n; i++)
	{
		E[i] = 1.0f - E[i];
	}

	// 物体の色と背景の色の、近くでの平均を、縮小した画像で求める
	const int s = (std::min(w, h) >= 400) ? 2 : 1;
	const int lw = (w + s - 1) / s;
	const int lh = (h + s - 1) / s;
	const size_t ln = (size_t)lw * lh;
	std::vector<float> lI(ln), lE(ln), fR(ln), fG(ln), fB(ln), bR(ln), bG(ln), bB(ln);
	parallel_for(lh, [&](int y_begin, int y_end)
		{
			for (int ly = y_begin; ly < y_end; ly++)
			{
				for (int lx = 0; lx < lw; lx++)
				{
					float sI = 0, sE = 0, sfr = 0, sfg = 0, sfb = 0, sbr = 0, sbg = 0, sbb = 0;
					int cnt = 0;
					for (int dy = 0; dy < s; dy++)
					{
						int y = ly * s + dy;
						if (y >= h)
						{
							break;
						}
						for (int dx = 0; dx < s; dx++)
						{
							int x = lx * s + dx;
							if (x >= w)
							{
								break;
							}
							size_t i = (size_t)y * w + x;
							float r = buf[i].r / 255.0f, g = buf[i].g / 255.0f, b = buf[i].b / 255.0f;
							sI += I[i];
							sE += E[i];
							sfr += r * I[i];
							sfg += g * I[i];
							sfb += b * I[i];
							sbr += r * E[i];
							sbg += g * E[i];
							sbb += b * E[i];
							cnt++;
						}
					}
					size_t li = (size_t)ly * lw + lx;
					float inv = cnt ? 1.0f / cnt : 0.0f;
					lI[li] = sI * inv;
					lE[li] = sE * inv;
					fR[li] = sfr * inv;
					fG[li] = sfg * inv;
					fB[li] = sfb * inv;
					bR[li] = sbr * inv;
					bG[li] = sbg * inv;
					bB[li] = sbb * inv;
				}
			}
		});
	const int rb = std::max(2, (int)std::ceil(band * 3.0f / s));
	std::vector<float> mI, mE, mfR, mfG, mfB, mbR, mbG, mbB;
	box_mean(lI, lw, lh, rb, mI);
	box_mean(lE, lw, lh, rb, mE);
	box_mean(fR, lw, lh, rb, mfR);
	box_mean(fG, lw, lh, rb, mfG);
	box_mean(fB, lw, lh, rb, mfB);
	box_mean(bR, lw, lh, rb, mbR);
	box_mean(bG, lw, lh, rb, mbG);
	box_mean(bB, lw, lh, rb, mbB);

	parallel_for(h, [&](int y_begin, int y_end)
		{
			for (int y = y_begin; y < y_end; y++)
			{
				float fy = (y + 0.5f) / s - 0.5f;
				int j0 = (int)std::floor(fy);
				float wy = fy - j0;
				int ja = std::clamp(j0, 0, lh - 1);
				int jb = std::clamp(j0 + 1, 0, lh - 1);
				for (int x = 0; x < w; x++)
				{
					size_t i = (size_t)y * w + x;
					if (I[i] > 0.999f || E[i] > 0.999f)
					{
						continue;	// 確実に物体、または確実に背景
					}
					float fx = (x + 0.5f) / s - 0.5f;
					int i0 = (int)std::floor(fx);
					float wx = fx - i0;
					int ia = std::clamp(i0, 0, lw - 1);
					int ib = std::clamp(i0 + 1, 0, lw - 1);
					auto bil = [&](const std::vector<float>& v)
						{
							return (v[(size_t)ja * lw + ia] * (1 - wx) + v[(size_t)ja * lw + ib] * wx) * (1 - wy)
								+ (v[(size_t)jb * lw + ia] * (1 - wx) + v[(size_t)jb * lw + ib] * wx) * wy;
						};
					float wi = bil(mI);
					float we = bil(mE);
					if (wi < 0.02f || we < 0.02f)
					{
						continue;	// 近くに物体か背景の色の見本が無い
					}
					float Fr = bil(mfR) / wi, Fg = bil(mfG) / wi, Fb = bil(mfB) / wi;
					float Br = bil(mbR) / we, Bg = bil(mbG) / we, Bb = bil(mbB) / we;
					float dr = Fr - Br, dg = Fg - Bg, db = Fb - Bb;
					float dd = dr * dr + dg * dg + db * db;
					if (dd < 0.02f)
					{
						continue;	// 物体と背景の色が近くて、混ざり具合を求められない
					}
					// 色の差が小さいほど、画像のノイズの影響が大きいので、元のマスクを残す割合を増やす
					float conf = std::clamp((dd - 0.02f) / 0.06f, 0.0f, 1.0f);
					float cr = buf[i].r / 255.0f - Br, cg = buf[i].g / 255.0f - Bg, cb = buf[i].b / 255.0f - Bb;
					float a = std::clamp((cr * dr + cg * dg + cb * db) / dd, 0.0f, 1.0f);
					a = q[i] * (1.0f - conf) + a * conf;
					q[i] = a;
					if (recolor && a > 0.01f && a < 0.98f)
					{
						buf[i].r = (unsigned char)std::clamp((int)std::lround(Fr * 255.0f), 0, 255);
						buf[i].g = (unsigned char)std::clamp((int)std::lround(Fg * 255.0f), 0, 255);
						buf[i].b = (unsigned char)std::clamp((int)std::lround(Fb * 255.0f), 0, 255);
					}
				}
			}
		});
}

// 色の均一化: 物体の中で、周りの色と大きく違う小さな点 (ゴミや重なった光など) を、周りの色で塗りなおす。
// 単色に近い物体 (文字やロゴなど) 向け。写真のような細かい模様のある物体では、模様も消えるので使わない
static void unify_colors(PIXEL_RGBA* buf, const std::vector<float>& vis, int w, int h, float radius)
{
	if (radius < 0.5f)
	{
		return;
	}
	const size_t n = (size_t)w * h;
	// 周りの色の平均は、縁から離れた確実な部分だけから求める
	std::vector<float> interior(vis);
	grow_alpha(interior, w, h, -2.0f);
	std::vector<float> cr(n), cg(n), cb(n);
	parallel_for(h, [&](int y_begin, int y_end)
		{
			for (int y = y_begin; y < y_end; y++)
			{
				for (int x = 0; x < w; x++)
				{
					size_t i = (size_t)y * w + x;
					float wi = interior[i];
					cr[i] = buf[i].r / 255.0f * wi;
					cg[i] = buf[i].g / 255.0f * wi;
					cb[i] = buf[i].b / 255.0f * wi;
				}
			}
		});
	const int r = std::max(3, (int)std::lround(radius * 2.0f));
	std::vector<float> wsum(interior);
	box_blur_full(cr, w, h, r);
	box_blur_full(cg, w, h, r);
	box_blur_full(cb, w, h, r);
	box_blur_full(wsum, w, h, r);
	const float T = 0.10f;
	parallel_for(h, [&](int y_begin, int y_end)
		{
			for (int y = y_begin; y < y_end; y++)
			{
				for (int x = 0; x < w; x++)
				{
					size_t i = (size_t)y * w + x;
					if (vis[i] <= 0.02f || wsum[i] < 0.05f)
					{
						continue;
					}
					float mr = cr[i] / wsum[i], mg = cg[i] / wsum[i], mb = cb[i] / wsum[i];
					float dr = buf[i].r / 255.0f - mr, dg = buf[i].g / 255.0f - mg, db = buf[i].b / 255.0f - mb;
					float d = std::sqrt(dr * dr + dg * dg + db * db);
					float k = std::clamp((d - T) / T, 0.0f, 1.0f);
					if (k <= 0.0f)
					{
						continue;
					}
					buf[i].r = (unsigned char)std::clamp((int)std::lround((buf[i].r / 255.0f * (1 - k) + mr * k) * 255.0f), 0, 255);
					buf[i].g = (unsigned char)std::clamp((int)std::lround((buf[i].g / 255.0f * (1 - k) + mg * k) * 255.0f), 0, 255);
					buf[i].b = (unsigned char)std::clamp((int)std::lround((buf[i].b / 255.0f * (1 - k) + mb * k) * 255.0f), 0, 255);
				}
			}
		});
}

// 透明度の切り詰め: lo 以下は完全に透明、hi 以上は完全に不透明にして、その間を滑らかにつなぐ。
// 物体のまわりに残る薄い半透明の層(もや)や、物体の中の薄くなった部分を取り除く
static void alpha_levels(std::vector<float>& q, int w, int h, float lo, float hi)
{
	lo = std::clamp(lo, 0.0f, 0.98f);
	hi = std::clamp(hi, lo + 0.02f, 1.0f);
	if (lo <= 0.0f && hi >= 1.0f)
	{
		return;
	}
	const float inv = 1.0f / (hi - lo);
	parallel_for(h, [&](int y_begin, int y_end)
		{
			for (int y = y_begin; y < y_end; y++)
			{
				float* row = q.data() + (size_t)y * w;
				for (int x = 0; x < w; x++)
				{
					row[x] = std::clamp((row[x] - lo) * inv, 0.0f, 1.0f);
				}
			}
		});
}

static void apply_mask_refined(PIXEL_RGBA* buf, int w, int h, const std::vector<float>& m, int mw, int mh,
	bool invert, float radius, float feather, float grow, float edge_color, float matte, float smooth_shape, float alpha_low, float alpha_high, float unify)
{
	// ぼかしを指定しているときは、縁を硬くしすぎない
	float gain = std::clamp(6.0f - feather * 0.3f, 1.5f, 6.0f);
	std::vector<float> q;
	if (radius >= 0.5f)
	{
		refine_alpha(buf, w, h, m, mw, mh, radius, gain, q);
	}
	else
	{
		// 補正なし: 拡大して、縁を少しはっきりさせるだけ
		upsample_mask(m, mw, mh, w, h, q);
		for (float& v : q)
		{
			v = std::clamp((v - 0.5f) * gain + 0.5f, 0.0f, 1.0f);
		}
	}
	grow_alpha(q, w, h, grow);
	matte_refine(buf, q, w, h, matte, !invert);
	if (smooth_shape >= 0.5f)
	{
		// 開く (細い突起を消す) → 閉じる (細いくぼみを埋める)
		grow_alpha(q, w, h, -smooth_shape);
		grow_alpha(q, w, h, smooth_shape);
		grow_alpha(q, w, h, smooth_shape);
		grow_alpha(q, w, h, -smooth_shape);
	}
	alpha_levels(q, w, h, alpha_low, alpha_high);
	if (edge_color >= 0.5f)
	{
		if (invert)
		{
			std::vector<float> visible(q.size());
			for (size_t i = 0; i < q.size(); i++)
			{
				visible[i] = 1.0f - q[i];
			}
			decontaminate_edges(buf, visible, w, h, edge_color);
		}
		else
		{
			decontaminate_edges(buf, q, w, h, edge_color);
		}
	}
	if (unify >= 0.5f && !invert)
	{
		unify_colors(buf, q, w, h, unify);
	}
	parallel_for(h, [&](int y_begin, int y_end)
		{
			for (int y = y_begin; y < y_end; y++)
			{
				PIXEL_RGBA* row = buf + (size_t)y * w;
				const float* qr = q.data() + (size_t)y * w;
				for (int x = 0; x < w; x++)
				{
					float v = invert ? 1.0f - qr[x] : qr[x];
					row[x].a = (unsigned char)std::clamp((int)std::lround(row[x].a * v), 0, 255);
				}
			}
		});
}

//---------------------------------------------------------------------
//	フレームの内容のハッシュ (静止画のように同じ内容のフレームを見分ける)
//---------------------------------------------------------------------
static uint64_t content_hash(const PIXEL_RGBA* img, int w, int h)
{
	uint64_t hash = 1469598103934665603ull;
	const int N = 64;
	for (int gy = 0; gy < N; gy++)
	{
		int y = std::min(h - 1, (int)(((int64_t)gy * 2 + 1) * h / (2 * N)));
		for (int gx = 0; gx < N; gx++)
		{
			int x = std::min(w - 1, (int)(((int64_t)gx * 2 + 1) * w / (2 * N)));
			const PIXEL_RGBA& p = img[(size_t)y * w + x];
			uint32_t v = (uint32_t)p.r | ((uint32_t)p.g << 8) | ((uint32_t)p.b << 16) | ((uint32_t)p.a << 24);
			hash ^= v;
			hash *= 1099511628211ull;
		}
	}
	return hash ? hash : 1;
}

//---------------------------------------------------------------------
//	フレームのシグネチャ
//---------------------------------------------------------------------
static void frame_signature(const PIXEL_RGBA* img, int w, int h, uint8_t* sig)
{
	for (int gy = 0; gy < 16; gy++)
	{
		int y = std::min(h - 1, (gy * 2 + 1) * h / 32);
		for (int gx = 0; gx < 16; gx++)
		{
			int x = std::min(w - 1, (gx * 2 + 1) * w / 32);
			const PIXEL_RGBA& p = img[(size_t)y * w + x];
			sig[gy * 16 + gx] = (uint8_t)((p.r * 77 + p.g * 150 + p.b * 29) >> 8);
		}
	}
}

static bool signature_match(const uint8_t* a, const uint8_t* b)
{
	int sum = 0;
	for (int i = 0; i < 256; i++)
	{
		sum += std::abs((int)a[i] - (int)b[i]);
	}
	return sum / 256 <= 6;
}

static uint64_t params_hash(const Params& p, int w, int h)
{
	uint64_t hash = 1469598103934665603ull;
	auto mix = [&hash](int v)
		{
			hash ^= (uint64_t)(uint32_t)v;
			hash *= 1099511628211ull;
		};
	mix(p.bx);
	mix(p.by);
	mix(p.bw);
	mix(p.bh);
	mix(p.anchor);
	for (const Shape& sh : p.shapes)
	{
		mix(sh.subtract ? 7 : 3);
		for (const auto& q : sh.pts)
		{
			mix(q.first);
			mix(q.second);
		}
	}
	mix(p.track ? 1 : 0);
	mix(p.interval);
	mix(w);
	mix(h);
	return hash;
}

static BoxF expand_box(const BoxF& b, int distance, int w, int h)
{
	float bw = b.x1 - b.x0;
	float bh = b.y1 - b.y0;
	float mx = bw * 0.08f + 6.0f + distance * 2.0f;
	float my = bh * 0.08f + 6.0f + distance * 2.0f;
	BoxF r;
	r.x0 = std::max(0.0f, b.x0 - mx);
	r.y0 = std::max(0.0f, b.y0 - my);
	r.x1 = std::min((float)w, b.x1 + mx);
	r.y1 = std::min((float)h, b.y1 + my);
	return r;
}

//---------------------------------------------------------------------
//	自由線で囲んだ範囲の解析とマスク化
//---------------------------------------------------------------------
// "x,y;x,y;..." 形式の文字列を解析する (元画像座標)
static void parse_polygon(LPCWSTR text, int w, int h, PolyPoints& poly)
{
	poly.clear();
	if (!text)
	{
		return;
	}
	const wchar_t* p = text;
	while (*p)
	{
		wchar_t* end = nullptr;
		long x = wcstol(p, &end, 10);
		if (end == p)
		{
			break;
		}
		p = end;
		if (*p != L',')
		{
			break;
		}
		p++;
		long y = wcstol(p, &end, 10);
		if (end == p)
		{
			break;
		}
		p = end;
		poly.push_back({ (int)std::clamp<long>(x, 0, w), (int)std::clamp<long>(y, 0, h) });
		while (*p == L';' || *p == L' ')
		{
			p++;
		}
	}
	if (poly.size() < 3)
	{
		poly.clear();
	}
}

// "+x,y;x,y;...|-x,y;x,y;..." 形式の文字列を解析する (+ は追加、- は除外。記号が無ければ追加)
static void parse_shapes(LPCWSTR text, int w, int h, ShapeList& shapes)
{
	shapes.clear();
	if (!text)
	{
		return;
	}
	std::wstring all = text;
	size_t pos = 0;
	while (pos <= all.size() && shapes.size() < 16)
	{
		size_t bar = all.find(L'|', pos);
		std::wstring part = all.substr(pos, bar == std::wstring::npos ? std::wstring::npos : bar - pos);
		bool subtract = false;
		if (!part.empty() && (part[0] == L'-' || part[0] == L'+'))
		{
			subtract = (part[0] == L'-');
			part.erase(0, 1);
		}
		Shape sh;
		sh.subtract = subtract;
		parse_polygon(part.c_str(), w, h, sh.pts);
		if (!sh.pts.empty())
		{
			shapes.push_back(std::move(sh));
		}
		if (bar == std::wstring::npos)
		{
			break;
		}
		pos = bar + 1;
	}
}

static bool shapes_have_add(const ShapeList& shapes)
{
	for (const Shape& sh : shapes)
	{
		if (!sh.subtract)
		{
			return true;
		}
	}
	return false;
}

static std::string shapes_to_string(const ShapeList& shapes)
{
	std::string out;
	for (size_t i = 0; i < shapes.size(); i++)
	{
		if (i)
		{
			out += '|';
		}
		out += shapes[i].subtract ? '-' : '+';
		for (const auto& q : shapes[i].pts)
		{
			char v[32];
			snprintf(v, sizeof(v), "%d,%d;", q.first, q.second);
			out += v;
		}
	}
	return out;
}

// 多角形を mw*mh のマスク(0/1)にする (偶奇規則)
static void rasterize_polygon(const PolyPoints& poly, int w, int h, int mw, int mh, std::vector<uint8_t>& out)
{
	out.assign((size_t)mw * mh, 0);
	const size_t n = poly.size();
	if (n < 3)
	{
		return;
	}
	std::vector<float> xs;
	for (int y = 0; y < mh; y++)
	{
		float py = (y + 0.5f) * h / mh;
		xs.clear();
		for (size_t i = 0; i < n; i++)
		{
			const auto& a = poly[i];
			const auto& b = poly[(i + 1) % n];
			float ay = (float)a.second;
			float by = (float)b.second;
			if ((ay <= py) != (by <= py))
			{
				float t = (py - ay) / (by - ay);
				xs.push_back(a.first + t * (b.first - a.first));
			}
		}
		std::sort(xs.begin(), xs.end());
		for (size_t k = 0; k + 1 < xs.size(); k += 2)
		{
			int x0 = (int)std::ceil(xs[k] * mw / w - 0.5f);
			int x1 = (int)std::floor(xs[k + 1] * mw / w - 0.5f);
			x0 = std::max(0, x0);
			x1 = std::min(mw - 1, x1);
			for (int x = x0; x <= x1; x++)
			{
				out[(size_t)y * mw + x] = 1;
			}
		}
	}
}

static void rasterize_rect(int bx, int by, int bw, int bh, int w, int h, int mw, int mh, std::vector<uint8_t>& out)
{
	out.assign((size_t)mw * mh, 0);
	for (int y = 0; y < mh; y++)
	{
		float py = (y + 0.5f) * h / mh;
		if (py < by || py > by + bh)
		{
			continue;
		}
		for (int x = 0; x < mw; x++)
		{
			float px = (x + 0.5f) * w / mw;
			if (px >= bx && px <= bx + bw)
			{
				out[(size_t)y * mw + x] = 1;
			}
		}
	}
}

// 領域(1)の各画素から、領域外(0)までの距離 (3-4 チャンファー距離。単位は 1/3 画素)
static void distance_transform(const std::vector<uint8_t>& region, int w, int h, std::vector<int>& d)
{
	const int INF = 1 << 28;
	d.assign((size_t)w * h, 0);
	for (size_t i = 0; i < d.size(); i++)
	{
		d[i] = region[i] ? INF : 0;
	}
	for (int y = 0; y < h; y++)
	{
		for (int x = 0; x < w; x++)
		{
			size_t i = (size_t)y * w + x;
			if (!d[i])
			{
				continue;
			}
			int v = d[i];
			if (x > 0)
			{
				v = std::min(v, d[i - 1] + 3);
			}
			if (y > 0)
			{
				v = std::min(v, d[i - w] + 3);
				if (x > 0)
				{
					v = std::min(v, d[i - w - 1] + 4);
				}
				if (x + 1 < w)
				{
					v = std::min(v, d[i - w + 1] + 4);
				}
			}
			d[i] = v;
		}
	}
	for (int y = h - 1; y >= 0; y--)
	{
		for (int x = w - 1; x >= 0; x--)
		{
			size_t i = (size_t)y * w + x;
			if (!d[i])
			{
				continue;
			}
			int v = d[i];
			if (x + 1 < w)
			{
				v = std::min(v, d[i + 1] + 3);
			}
			if (y + 1 < h)
			{
				v = std::min(v, d[i + w] + 3);
				if (x + 1 < w)
				{
					v = std::min(v, d[i + w + 1] + 4);
				}
				if (x > 0)
				{
					v = std::min(v, d[i + w - 1] + 4);
				}
			}
			d[i] = v;
		}
	}
}

// 領域の内側で、縁から遠く、かつ互いに離れた点を最大 count 個選ぶ (マスク座標)。min_depth は 1/3 画素単位
static void pick_points(const std::vector<uint8_t>& region, int w, int h, int count, int min_depth, std::vector<std::pair<float, float>>& out)
{
	std::vector<int> d;
	distance_transform(region, w, h, d);
	int maxd = 0;
	size_t best = 0;
	for (size_t i = 0; i < d.size(); i++)
	{
		if (d[i] > maxd)
		{
			maxd = d[i];
			best = i;
		}
	}
	if (maxd < min_depth || maxd >= (1 << 27))
	{
		return;
	}
	std::vector<std::pair<int, int>> chosen;
	chosen.push_back({ (int)(best % w), (int)(best / w) });
	int thr = std::max(min_depth, (int)(maxd * 0.4f));
	for (int k = 1; k < count; k++)
	{
		int64_t best_dist = -1;
		std::pair<int, int> best_pt = { 0, 0 };
		for (int y = 0; y < h; y++)
		{
			for (int x = 0; x < w; x++)
			{
				if (d[(size_t)y * w + x] < thr)
				{
					continue;
				}
				int64_t nearest = INT64_MAX;
				for (const auto& c : chosen)
				{
					int64_t dx = x - c.first;
					int64_t dy = y - c.second;
					nearest = std::min(nearest, dx * dx + dy * dy);
				}
				if (nearest > best_dist)
				{
					best_dist = nearest;
					best_pt = { x, y };
				}
			}
		}
		if (best_dist < 36)
		{
			break;
		}
		chosen.push_back(best_pt);
	}
	for (const auto& c : chosen)
	{
		out.push_back({ c.first + 0.5f, c.second + 0.5f });
	}
}

// 画像を mw*mh に縮小した RGB (3バイト/画素)。点を選ぶための簡易な縮小
static void make_small_rgb(const PIXEL_RGBA* img, int w, int h, int mw, int mh, std::vector<uint8_t>& out)
{
	out.assign((size_t)mw * mh * 3, 0);
	parallel_for(mh, [&](int y_begin, int y_end)
		{
			for (int y = y_begin; y < y_end; y++)
			{
				int sy = std::min(h - 1, (int)((y + 0.5f) * h / mh));
				for (int x = 0; x < mw; x++)
				{
					int sx = std::min(w - 1, (int)((x + 0.5f) * w / mw));
					const PIXEL_RGBA& p = img[(size_t)sy * w + sx];
					size_t o = ((size_t)y * mw + x) * 3;
					out[o] = p.r;
					out[o + 1] = p.g;
					out[o + 2] = p.b;
				}
			}
		});
}

// 囲んだ範囲(seed)の外側の色と違う部分を「物体上の点」、範囲の縁のすぐ外側を「背景上の点」として選ぶ。
// 外側と同じような色しか無く、物体を見分けられない場合は false (従来の方法を使う)
static bool pick_points_by_color(const std::vector<uint8_t>& seed, const std::vector<uint8_t>& rgb, int mw, int mh,
	int pos_count, int neg_count, std::vector<std::pair<float, float>>& pos, std::vector<std::pair<float, float>>& neg)
{
	const size_t n = (size_t)mw * mh;
	const int UNUSABLE = 1 << 27;
	std::vector<int> din;
	distance_transform(seed, mw, mh, din);
	std::vector<uint8_t> outside(n);
	int minx = mw, miny = mh, maxx = -1, maxy = -1;
	for (int y = 0; y < mh; y++)
	{
		for (int x = 0; x < mw; x++)
		{
			size_t i = (size_t)y * mw + x;
			outside[i] = seed[i] ? 0 : 1;
			if (seed[i])
			{
				minx = std::min(minx, x);
				miny = std::min(miny, y);
				maxx = std::max(maxx, x);
				maxy = std::max(maxy, y);
			}
		}
	}
	if (maxx < 0)
	{
		return false;
	}
	std::vector<int> dout;
	distance_transform(outside, mw, mh, dout);

	// 範囲の外側の帯 (縁から 2 画素以上離れ、範囲の大きさに応じた幅まで) を背景の色見本にする
	int ring_px = std::clamp((int)(0.08f * std::max(maxx - minx, maxy - miny)), 6, 40);
	std::vector<std::pair<int, int>> ring;
	for (int y = 0; y < mh; y++)
	{
		for (int x = 0; x < mw; x++)
		{
			int d = dout[(size_t)y * mw + x];
			if (d >= 6 && d <= ring_px * 3 && d < UNUSABLE)
			{
				ring.push_back({ x, y });
			}
		}
	}
	if (ring.size() < 50)
	{
		return false;
	}
	size_t stride = std::max<size_t>(1, ring.size() / 600);
	std::vector<std::array<int, 3>> samples;
	for (size_t i = 0; i < ring.size(); i += stride)
	{
		size_t o = ((size_t)ring[i].second * mw + ring[i].first) * 3;
		samples.push_back({ rgb[o], rgb[o + 1], rgb[o + 2] });
	}

	// 範囲の内側で、背景の色見本のどれとも違う画素 (= 物体らしい画素) を探す
	const int T2 = 40 * 40;		// これより離れていれば「どの背景の色とも違う」
	const int T2_SAME = 18 * 18;	// これより近ければ「背景とそっくり」
	struct Cand
	{
		int x, y, depth;
	};
	std::vector<Cand> cands;
	std::vector<uint8_t> bgmap(n, 0);	// 範囲の内側の画素: 0 = 背景と違う色, 1 = 背景に近い色, 2 = 背景とそっくり
	for (int y = 0; y < mh; y++)
	{
		for (int x = 0; x < mw; x++)
		{
			size_t i = (size_t)y * mw + x;
			if (!seed[i] || din[i] >= UNUSABLE)
			{
				continue;
			}
			size_t o = i * 3;
			bool like_bg = false;
			bool same_bg = false;
			for (const auto& c : samples)
			{
				int dr = rgb[o] - c[0];
				int dg = rgb[o + 1] - c[1];
				int db = rgb[o + 2] - c[2];
				int d2 = dr * dr + dg * dg + db * db;
				if (d2 < T2_SAME)
				{
					like_bg = true;
					same_bg = true;
					break;
				}
				if (d2 < T2)
				{
					like_bg = true;
				}
			}
			bgmap[i] = same_bg ? 2 : (like_bg ? 1 : 0);
			if (!like_bg && din[i] >= 6)
			{
				cands.push_back({ x, y, din[i] });
			}
		}
	}
	if (cands.size() < 20)
	{
		return false;
	}

	// 範囲の縁から背景に近い色でつながっている部分を、外側の背景として塗りつぶす。
	// つながっていない「背景とそっくりな領域」が、物体に囲まれた穴 (文字の穴の中など)
	std::vector<uint8_t> reached(n, 0);
	std::vector<size_t> queue;
	for (size_t i = 0; i < n; i++)
	{
		if (seed[i] && bgmap[i] >= 1 && din[i] < 6)
		{
			reached[i] = 1;
			queue.push_back(i);
		}
	}
	for (size_t head = 0; head < queue.size(); head++)
	{
		size_t i = queue[head];
		int x = (int)(i % mw);
		int y = (int)(i / mw);
		const int nx[4] = { x - 1, x + 1, x, x };
		const int ny[4] = { y, y, y - 1, y + 1 };
		for (int k = 0; k < 4; k++)
		{
			if (nx[k] < 0 || nx[k] >= mw || ny[k] < 0 || ny[k] >= mh)
			{
				continue;
			}
			size_t j = (size_t)ny[k] * mw + nx[k];
			if (!reached[j] && seed[j] && bgmap[j] >= 1)
			{
				reached[j] = 1;
				queue.push_back(j);
			}
		}
	}
	std::vector<uint8_t> hole_region(n, 0);
	for (size_t i = 0; i < n; i++)
	{
		if (seed[i] && bgmap[i] == 2 && !reached[i])
		{
			hole_region[i] = 1;
		}
	}

	// 物体上の点: まず範囲の縁から最も遠い候補、次に既に選んだ点から離れた候補
	std::vector<std::pair<int, int>> chosen;
	size_t best = 0;
	for (size_t i = 1; i < cands.size(); i++)
	{
		if (cands[i].depth > cands[best].depth)
		{
			best = i;
		}
	}
	chosen.push_back({ cands[best].x, cands[best].y });
	for (int k = 1; k < pos_count; k++)
	{
		int64_t best_dist = -1;
		std::pair<int, int> best_pt = { 0, 0 };
		for (const Cand& c : cands)
		{
			int64_t nearest = INT64_MAX;
			for (const auto& q : chosen)
			{
				int64_t dx = c.x - q.first;
				int64_t dy = c.y - q.second;
				nearest = std::min(nearest, dx * dx + dy * dy);
			}
			if (nearest > best_dist)
			{
				best_dist = nearest;
				best_pt = { c.x, c.y };
			}
		}
		if (best_dist < 49)
		{
			break;
		}
		chosen.push_back(best_pt);
	}
	for (const auto& q : chosen)
	{
		pos.push_back({ q.first + 0.5f, q.second + 0.5f });
	}

	// 背景上の点(1): 物体に囲まれた、背景とそっくりな広い領域 (文字の穴の中など)。細い線や小さな点は除く
	{
		std::vector<std::pair<float, float>> holes;
		pick_points(hole_region, mw, mh, 3, 9, holes);
		for (const auto& q : holes)
		{
			// 物体上の点の近くは避ける
			bool near_pos = false;
			for (const auto& c : chosen)
			{
				float dx = q.first - (c.first + 0.5f);
				float dy = q.second - (c.second + 0.5f);
				if (dx * dx + dy * dy < 64.0f)
				{
					near_pos = true;
					break;
				}
			}
			if (!near_pos)
			{
				neg.push_back(q);
			}
		}
	}

	// 背景上の点(2): 範囲の縁のすぐ外側 (2〜8画素) から、互いに離れた点
	std::vector<std::pair<int, int>> band;
	for (const auto& r : ring)
	{
		int d = dout[(size_t)r.second * mw + r.first];
		if (d <= 8 * 3)
		{
			band.push_back(r);
		}
	}
	if (band.empty())
	{
		return true;
	}
	std::vector<std::pair<int, int>> nchosen;
	nchosen.push_back(band[0]);
	for (int k = 1; k < neg_count; k++)
	{
		int64_t best_dist = -1;
		std::pair<int, int> best_pt = band[0];
		for (const auto& r : band)
		{
			int64_t nearest = INT64_MAX;
			for (const auto& q : nchosen)
			{
				int64_t dx = r.first - q.first;
				int64_t dy = r.second - q.second;
				nearest = std::min(nearest, dx * dx + dy * dy);
			}
			if (nearest > best_dist)
			{
				best_dist = nearest;
				best_pt = r;
			}
		}
		if (best_dist < 144)
		{
			break;
		}
		nchosen.push_back(best_pt);
	}
	for (const auto& q : nchosen)
	{
		neg.push_back({ q.first + 0.5f, q.second + 0.5f });
	}
	return true;
}

//---------------------------------------------------------------------
//	軽量な追跡 (SAM2を毎フレーム動かさず、物体の移動量だけを推定してマスクをずらす)
//---------------------------------------------------------------------
static void compute_thumb(const PIXEL_RGBA* img, int w, int h, int tw, int th, std::vector<uint8_t>& out)
{
	out.assign((size_t)tw * th, 0);
	parallel_for(th, [&](int y_begin, int y_end)
		{
			for (int ty = y_begin; ty < y_end; ty++)
			{
				int by = ty * h / th;
				int sh = std::max(1, h / th);
				int ya = std::min(h - 1, by + sh / 4);
				int yb = std::min(h - 1, by + sh * 3 / 4);
				for (int tx = 0; tx < tw; tx++)
				{
					int bx = tx * w / tw;
					int sw = std::max(1, w / tw);
					int xa = std::min(w - 1, bx + sw / 4);
					int xb = std::min(w - 1, bx + sw * 3 / 4);
					const PIXEL_RGBA& p0 = img[(size_t)ya * w + xa];
					const PIXEL_RGBA& p1 = img[(size_t)ya * w + xb];
					const PIXEL_RGBA& p2 = img[(size_t)yb * w + xa];
					const PIXEL_RGBA& p3 = img[(size_t)yb * w + xb];
					int sum = ((p0.r * 77 + p0.g * 150 + p0.b * 29) >> 8) + ((p1.r * 77 + p1.g * 150 + p1.b * 29) >> 8)
						+ ((p2.r * 77 + p2.g * 150 + p2.b * 29) >> 8) + ((p3.r * 77 + p3.g * 150 + p3.b * 29) >> 8);
					out[(size_t)ty * tw + tx] = (uint8_t)(sum / 4);
				}
			}
		});
}

// 前フレームの物体の画素が、現在のフレームのどこへ動いたかを (サムネイル解像度で) 探す
static bool estimate_shift(const MaskEntry& prev, const std::vector<uint8_t>& cur_thumb, int tw, int th, int gap, int& dx, int& dy, float& err)
{
	std::vector<std::pair<int, int>> fg;
	for (int ty = 0; ty < th; ty++)
	{
		int my = std::min(prev.h - 1, ty * prev.h / th);
		for (int tx = 0; tx < tw; tx++)
		{
			int mx = std::min(prev.w - 1, tx * prev.w / tw);
			if (prev.a[(size_t)my * prev.w + mx] >= 128)
			{
				fg.push_back({ tx, ty });
			}
		}
	}
	if (fg.size() < 30)
	{
		return false;
	}
	// 計算量を抑えるため、最大 3000 点に間引く
	size_t stride = std::max<size_t>(1, fg.size() / 3000);
	std::vector<std::pair<int, int>> pts;
	for (size_t i = 0; i < fg.size(); i += stride)
	{
		pts.push_back(fg[i]);
	}

	const int R = std::min(tw / 8, std::max(8, tw / 25 * gap));	// 探す範囲 (サムネイル画素)。間隔が空くほど広げる
	float best = 1e30f;
	int best_dx = 0, best_dy = 0;
	float best_mean = 1e30f;
	for (int sy = -R; sy <= R; sy++)
	{
		for (int sx = -R; sx <= R; sx++)
		{
			int sum = 0;
			int n = 0;
			for (const auto& q : pts)
			{
				int nx = q.first + sx;
				int ny = q.second + sy;
				if (nx < 0 || nx >= tw || ny < 0 || ny >= th)
				{
					continue;
				}
				sum += std::abs((int)prev.thumb[(size_t)q.second * tw + q.first] - (int)cur_thumb[(size_t)ny * tw + nx]);
				n++;
			}
			if (n < (int)(pts.size() * 6 / 10))
			{
				continue;
			}
			float mean = (float)sum / n;
			float score = mean + 0.02f * (std::abs(sx) + std::abs(sy));	// 同点なら小さい移動を優先
			if (score < best)
			{
				best = score;
				best_mean = mean;
				best_dx = sx;
				best_dy = sy;
			}
		}
	}
	if (best_mean > 1e29f)
	{
		return false;
	}
	dx = best_dx;
	dy = best_dy;
	err = best_mean;
	return true;
}

static void translate_mask(const std::vector<uint8_t>& src, int w, int h, int sx, int sy, std::vector<uint8_t>& dst)
{
	dst.assign(src.size(), 0);
	for (int y = 0; y < h; y++)
	{
		int yy = y - sy;
		if (yy < 0 || yy >= h)
		{
			continue;
		}
		for (int x = 0; x < w; x++)
		{
			int xx = x - sx;
			if (xx >= 0 && xx < w)
			{
				dst[(size_t)y * w + x] = src[(size_t)yy * w + xx];
			}
		}
	}
}

static int count_mask(const std::vector<uint8_t>& a)
{
	int n = 0;
	for (uint8_t v : a)
	{
		n += (v >= 128) ? 1 : 0;
	}
	return n;
}

// 近くのキャッシュ済みフレームから、マスクをずらして今のフレームのマスクを作る。できなければ false
static bool try_propagate(CacheBucket* inst, const Params& p, int frame, int w, int h, int mw, int mh,
	int tw, int th, const std::vector<uint8_t>& cur_thumb, const uint8_t* cur_sig, MaskEntry& out)
{
	if (!p.track || p.interval <= 1 || frame == p.anchor)
	{
		return false;
	}
	for (int g = 1; g <= 3; g++)
	{
		auto it = inst->masks.find(frame - g);
		if (it == inst->masks.end())
		{
			continue;
		}
		const MaskEntry& pe = it->second;
		if (!pe.valid || pe.w != mw || pe.h != mh || pe.tw != tw || pe.th != th || pe.thumb.empty() || pe.a.empty())
		{
			continue;
		}
		if (pe.age + g >= p.interval)
		{
			return false;	// 検出し直す時期
		}
		int dx = 0, dy = 0;
		float err = 0.0f;
		if (!estimate_shift(pe, cur_thumb, tw, th, g, dx, dy, err) || err > 28.0f)
		{
			return false;	// 動きが推定できない (急な変化) ので検出し直す
		}
		int sx = (int)std::lround((float)dx * mw / tw);
		int sy = (int)std::lround((float)dy * mh / th);
		MaskEntry e;
		e.w = mw;
		e.h = mh;
		translate_mask(pe.a, mw, mh, sx, sy, e.a);
		int before = count_mask(pe.a);
		int after = count_mask(e.a);
		if (after < before * 7 / 10)
		{
			return false;	// 物体が画面外へ出ていく場合などは検出し直す
		}
		e.valid = true;
		float fx = (float)sx * w / mw;
		float fy = (float)sy * h / mh;
		e.bbox.x0 = std::clamp(pe.bbox.x0 + fx, 0.0f, (float)w);
		e.bbox.x1 = std::clamp(pe.bbox.x1 + fx, 0.0f, (float)w);
		e.bbox.y0 = std::clamp(pe.bbox.y0 + fy, 0.0f, (float)h);
		e.bbox.y1 = std::clamp(pe.bbox.y1 + fy, 0.0f, (float)h);
		memcpy(e.sig, cur_sig, sizeof(e.sig));
		e.thumb = cur_thumb;
		e.tw = tw;
		e.th = th;
		e.age = pe.age + g;
		out = std::move(e);
		return true;
	}
	return false;
}

static void evict_cache(CacheBucket* inst, int frame)
{
	while (inst->masks.size() > CACHE_MAX_FRAMES)
	{
		auto first = inst->masks.begin();
		auto last = std::prev(inst->masks.end());
		if (std::abs(first->first - frame) >= std::abs(last->first - frame))
		{
			inst->masks.erase(first);
		}
		else
		{
			inst->masks.erase(last);
		}
	}
}

//---------------------------------------------------------------------
//	マスクキャッシュ (メモリ + ファイル)
//---------------------------------------------------------------------
static const size_t MAX_BUCKETS = 8;
static std::mutex g_cache_mtx;
static std::mutex g_file_mtx;
static std::map<uint64_t, std::unique_ptr<CacheBucket>> g_buckets;
static int g_cache_generation = 0;

static void rle_encode(const std::vector<uint8_t>& src, std::vector<uint8_t>& out)
{
	out.clear();
	size_t n = src.size();
	size_t i = 0;
	while (i < n)
	{
		uint8_t v = src[i];
		size_t j = i + 1;
		while (j < n && src[j] == v && j - i < 65535)
		{
			j++;
		}
		size_t run = j - i;
		out.push_back((uint8_t)(run & 0xff));
		out.push_back((uint8_t)(run >> 8));
		out.push_back(v);
		i = j;
	}
}

static bool rle_decode(const uint8_t* p, size_t n, size_t expect, std::vector<uint8_t>& out)
{
	out.clear();
	out.reserve(expect);
	for (size_t i = 0; i + 3 <= n; i += 3)
	{
		size_t run = (size_t)p[i] | ((size_t)p[i + 1] << 8);
		if (out.size() + run > expect)
		{
			out.clear();
			return false;
		}
		out.insert(out.end(), run, p[i + 2]);
	}
	if (out.size() != expect)
	{
		out.clear();
		return false;
	}
	return true;
}

// キャッシュのエントリを、マスク本体(a)つきのコピーとして取り出す
static bool load_hot(const MaskEntry& src, MaskEntry& dst)
{
	dst = src;
	if (dst.a.empty())
	{
		if (!rle_decode(src.z.data(), src.z.size(), (size_t)src.w * src.h, dst.a))
		{
			return false;
		}
	}
	return true;
}

static void mask_dims(int w, int h, int& mw, int& mh, int& tw, int& th)
{
	float s = std::min(1.0f, (float)MASK_MAX_SIZE / (float)std::max(w, h));
	mw = std::max(1, (int)std::lround(w * s));
	mh = std::max(1, (int)std::lround(h * s));
	tw = TRACK_THUMB_W;
	th = std::max(1, (int)std::lround((float)h * tw / w));
}

// エントリを保存する。現在のフレームの近く以外は、マスク本体とサムネイルを手放してメモリを節約する
static void store_entry(CacheBucket* b, int frame, MaskEntry&& e)
{
	rle_encode(e.a, e.z);
	uint64_t ch = e.chash;
	b->masks[frame] = std::move(e);
	if (ch)
	{
		b->by_hash[ch] = frame;
	}
	b->pending.push_back(frame);
	for (auto& kv : b->masks)
	{
		if (std::abs(kv.first - frame) > 4 && !kv.second.a.empty())
		{
			std::vector<uint8_t>().swap(kv.second.a);
			std::vector<uint8_t>().swap(kv.second.thumb);
		}
	}
	evict_cache(b, frame);
}

static std::wstring cache_dir()
{
	return plugin_dir() + L"\\ObjectMaskCache";
}

static std::wstring cache_path(uint64_t key)
{
	static const wchar_t* hex = L"0123456789abcdef";
	std::wstring name;
	for (int i = 15; i >= 0; i--)
	{
		name += hex[(key >> (i * 4)) & 0xf];
	}
	return cache_dir() + L"\\" + name + L".omc";
}

static void put_u32(std::vector<uint8_t>& b, uint32_t v)
{
	for (int i = 0; i < 4; i++)
	{
		b.push_back((uint8_t)(v >> (8 * i)));
	}
}

static void put_f32(std::vector<uint8_t>& b, float f)
{
	uint32_t v;
	memcpy(&v, &f, 4);
	put_u32(b, v);
}

static uint32_t get_u32(const uint8_t* p)
{
	return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static float get_f32(const uint8_t* p)
{
	uint32_t v = get_u32(p);
	float f;
	memcpy(&f, &v, 4);
	return f;
}

static const size_t RECORD_FIXED = 4 + 1 + 4 + 16 + 256 + 8 + 4;	// 1フレーム分の記録の、圧縮データより前の部分

static void serialize_entry(std::vector<uint8_t>& b, int frame, const MaskEntry& e)
{
	put_u32(b, (uint32_t)frame);
	b.push_back(e.valid ? 1 : 0);
	put_u32(b, (uint32_t)e.age);
	put_f32(b, e.bbox.x0);
	put_f32(b, e.bbox.y0);
	put_f32(b, e.bbox.x1);
	put_f32(b, e.bbox.y1);
	b.insert(b.end(), e.sig, e.sig + 256);
	put_u32(b, (uint32_t)(e.chash & 0xffffffffu));
	put_u32(b, (uint32_t)(e.chash >> 32));
	put_u32(b, (uint32_t)e.z.size());
	b.insert(b.end(), e.z.begin(), e.z.end());
}

static void delete_cache_files()
{
	std::wstring dir = cache_dir();
	WIN32_FIND_DATAW fd;
	HANDLE h = FindFirstFileW((dir + L"\\*.omc").c_str(), &fd);
	if (h == INVALID_HANDLE_VALUE)
	{
		return;
	}
	do
	{
		DeleteFileW((dir + L"\\" + fd.cFileName).c_str());
	} while (FindNextFileW(h, &fd));
	FindClose(h);
}

// 古いキャッシュファイルを削除する。2週間以上使われていないもの、および合計が上限を超えた分 (古い順)
static const ULONGLONG CACHE_MAX_BYTES = 1024ull * 1024 * 1024;	// ファイル全体の上限 (1GB)

static void cleanup_old_cache()
{
	// 使用中のキャッシュは消さない
	std::vector<std::wstring> active;
	{
		std::lock_guard<std::mutex> lock(g_cache_mtx);
		for (auto& kv : g_buckets)
		{
			active.push_back(cache_path(kv.first));
		}
	}
	struct FileInfo
	{
		std::wstring path;
		ULONGLONG time;
		ULONGLONG size;
	};
	std::vector<FileInfo> files;
	std::wstring dir = cache_dir();
	WIN32_FIND_DATAW fd;
	HANDLE h = FindFirstFileW((dir + L"\\*.omc").c_str(), &fd);
	if (h == INVALID_HANDLE_VALUE)
	{
		return;
	}
	do
	{
		FileInfo fi;
		fi.path = dir + L"\\" + fd.cFileName;
		ULARGE_INTEGER t;
		t.LowPart = fd.ftLastWriteTime.dwLowDateTime;
		t.HighPart = fd.ftLastWriteTime.dwHighDateTime;
		fi.time = t.QuadPart;
		fi.size = ((ULONGLONG)fd.nFileSizeHigh << 32) | fd.nFileSizeLow;
		if (std::find(active.begin(), active.end(), fi.path) == active.end())
		{
			files.push_back(fi);
		}
	} while (FindNextFileW(h, &fd));
	FindClose(h);

	FILETIME now_ft;
	GetSystemTimeAsFileTime(&now_ft);
	ULARGE_INTEGER now;
	now.LowPart = now_ft.dwLowDateTime;
	now.HighPart = now_ft.dwHighDateTime;
	const ULONGLONG limit = 14ull * 24 * 3600 * 10000000ull;

	// 2週間以上前のものを削除
	std::vector<FileInfo> kept;
	ULONGLONG total = 0;
	for (const FileInfo& fi : files)
	{
		if (now.QuadPart > fi.time && now.QuadPart - fi.time > limit)
		{
			DeleteFileW(fi.path.c_str());
		}
		else
		{
			kept.push_back(fi);
			total += fi.size;
		}
	}
	// 合計が上限を超えていれば、古い順に削除
	std::sort(kept.begin(), kept.end(), [](const FileInfo& a, const FileInfo& b) { return a.time < b.time; });
	for (const FileInfo& fi : kept)
	{
		if (total <= CACHE_MAX_BYTES)
		{
			break;
		}
		DeleteFileW(fi.path.c_str());
		total -= fi.size;
	}
}

// ファイルを開く (開けなければ nullptr)
static FILE* open_file(const std::wstring& path, const wchar_t* mode)
{
	FILE* f = nullptr;
	if (_wfopen_s(&f, path.c_str(), mode) != 0)
	{
		return nullptr;
	}
	return f;
}

static void load_from_disk(CacheBucket* b)
{
	std::wstring path = cache_path(b->key);
	FILE* f = open_file(path, L"rb");
	if (!f)
	{
		return;
	}
	std::vector<uint8_t> data;
	fseek(f, 0, SEEK_END);
	long size = ftell(f);
	fseek(f, 0, SEEK_SET);
	if (size > 0)
	{
		data.resize((size_t)size);
		if (fread(data.data(), 1, data.size(), f) != data.size())
		{
			data.clear();
		}
	}
	fclose(f);
	if (data.size() < 12 || memcmp(data.data(), "OMC3", 4) != 0
		|| (int)get_u32(&data[4]) != b->mw || (int)get_u32(&data[8]) != b->mh)
	{
		DeleteFileW(path.c_str());
		return;
	}
	size_t pos = 12;
	while (pos + RECORD_FIXED <= data.size())
	{
		const uint8_t* r = &data[pos];
		size_t zlen = get_u32(r + 289);
		if (pos + RECORD_FIXED + zlen > data.size())
		{
			break;
		}
		MaskEntry e;
		e.w = b->mw;
		e.h = b->mh;
		int frame = (int)get_u32(r);
		e.valid = r[4] != 0;
		e.age = (int)get_u32(r + 5);
		e.bbox.x0 = get_f32(r + 9);
		e.bbox.y0 = get_f32(r + 13);
		e.bbox.x1 = get_f32(r + 17);
		e.bbox.y1 = get_f32(r + 21);
		memcpy(e.sig, r + 25, 256);
		e.chash = (uint64_t)get_u32(r + 281) | ((uint64_t)get_u32(r + 285) << 32);
		e.z.assign(r + RECORD_FIXED, r + RECORD_FIXED + zlen);
		if (e.chash)
		{
			b->by_hash[e.chash] = frame;
		}
		b->masks[frame] = std::move(e);
		pos += RECORD_FIXED + zlen;
	}
	b->file_ready = true;
}

struct FlushJob
{
	std::wstring path;
	bool write_header = false;
	int mw = 0, mh = 0;
	std::vector<uint8_t> data;
};

// g_cache_mtx をロックした状態で呼ぶ。未保存のフレームを書き出し用のデータにまとめる
static void collect_jobs_locked(CacheBucket* b, std::vector<FlushJob>& jobs)
{
	if (b->pending.empty())
	{
		return;
	}
	std::sort(b->pending.begin(), b->pending.end());
	b->pending.erase(std::unique(b->pending.begin(), b->pending.end()), b->pending.end());
	FlushJob j;
	j.path = cache_path(b->key);
	j.write_header = !b->file_ready;
	j.mw = b->mw;
	j.mh = b->mh;
	for (int frame : b->pending)
	{
		auto it = b->masks.find(frame);
		if (it != b->masks.end() && !it->second.z.empty())
		{
			serialize_entry(j.data, frame, it->second);
		}
	}
	b->pending.clear();
	b->file_ready = true;
	jobs.push_back(std::move(j));
}

static void write_jobs(std::vector<FlushJob>& jobs)
{
	if (jobs.empty())
	{
		return;
	}
	std::lock_guard<std::mutex> lock(g_file_mtx);
	CreateDirectoryW(cache_dir().c_str(), nullptr);
	for (auto& j : jobs)
	{
		FILE* f = open_file(j.path, j.write_header ? L"wb" : L"ab");
		if (!f)
		{
			continue;
		}
		if (j.write_header)
		{
			std::vector<uint8_t> head = { 'O', 'M', 'C', '3' };
			put_u32(head, (uint32_t)j.mw);
			put_u32(head, (uint32_t)j.mh);
			fwrite(head.data(), 1, head.size(), f);
		}
		if (!j.data.empty())
		{
			fwrite(j.data.data(), 1, j.data.size(), f);
		}
		fclose(f);
	}
}

static void flush_pending_all()
{
	std::vector<FlushJob> jobs;
	{
		std::lock_guard<std::mutex> lock(g_cache_mtx);
		for (auto& kv : g_buckets)
		{
			collect_jobs_locked(kv.second.get(), jobs);
		}
	}
	write_jobs(jobs);
}

// g_cache_mtx をロックした状態で呼ぶ
static CacheBucket* acquire_bucket(const Params& p, int w, int h)
{
	int gen = g_generation.load();
	if (gen != g_cache_generation)
	{
		// 「解析キャッシュを破棄」が押された
		g_buckets.clear();
		delete_cache_files();
		g_cache_generation = gen;
	}
	uint64_t key = params_hash(p, w, h);
	auto it = g_buckets.find(key);
	if (it == g_buckets.end())
	{
		if (g_buckets.size() >= MAX_BUCKETS)
		{
			// 最も使われていないものを、未保存分を書き出してから捨てる
			auto oldest = g_buckets.begin();
			for (auto i = g_buckets.begin(); i != g_buckets.end(); ++i)
			{
				if (i->second->last_used < oldest->second->last_used)
				{
					oldest = i;
				}
			}
			std::vector<FlushJob> jobs;
			collect_jobs_locked(oldest->second.get(), jobs);
			write_jobs(jobs);
			g_buckets.erase(oldest);
		}
		std::unique_ptr<CacheBucket> b(new CacheBucket());
		b->key = key;
		int tw, th;
		mask_dims(w, h, b->mw, b->mh, tw, th);
		load_from_disk(b.get());
		it = g_buckets.emplace(key, std::move(b)).first;
	}
	it->second->last_used = GetTickCount64();
	return it->second.get();
}

static std::thread g_flush_thread;
static std::atomic<bool> g_flush_stop{ false };

static void flush_loop()
{
	cleanup_old_cache();
	int rounds = 0;
	while (!g_flush_stop.load())
	{
		for (int i = 0; i < 30 && !g_flush_stop.load(); i++)
		{
			Sleep(100);
		}
		flush_pending_all();
		if (++rounds >= 100)	// 約5分ごとに古いキャッシュを整理する
		{
			rounds = 0;
			cleanup_old_cache();
		}
	}
	flush_pending_all();
}

//---------------------------------------------------------------------
//	マスクの取得 (キャッシュになければ推論する)。g_cache_mtx をロックした状態で呼ぶ
//---------------------------------------------------------------------
static bool get_mask(CacheBucket* inst, const Params& p, int frame, const PIXEL_RGBA* buf, int w, int h, uint64_t chash,
	MaskEntry& cur, MaskEntry& prev, bool& has_prev, bool& content_hit, std::wstring& status)
{
	uint8_t cur_sig[256];
	frame_signature(buf, w, h, cur_sig);

	bool hit = false;
	auto it = inst->masks.find(frame);
	if (it != inst->masks.end() && signature_match(it->second.sig, cur_sig) && load_hot(it->second, cur))
	{
		hit = true;
	}
	content_hit = false;
	if (!hit)
	{
		// フレーム番号が違っても、内容が同じ(静止画など)なら、解析済みのマスクを使う
		auto hi = inst->by_hash.find(chash);
		if (hi != inst->by_hash.end())
		{
			auto e2 = inst->masks.find(hi->second);
			if (e2 != inst->masks.end() && e2->second.chash == chash && signature_match(e2->second.sig, cur_sig) && load_hot(e2->second, cur))
			{
				hit = true;
				content_hit = true;
			}
		}
	}

	int mw, mh, tw, th;
	mask_dims(w, h, mw, mh, tw, th);
	std::vector<uint8_t> cur_thumb;

	if (!hit)
	{
		compute_thumb(buf, w, h, tw, th, cur_thumb);
		MaskEntry moved;
		if (try_propagate(inst, p, frame, w, h, mw, mh, tw, th, cur_thumb, cur_sig, moved))
		{
			moved.chash = chash;
			cur = moved;
			store_entry(inst, frame, std::move(moved));
			hit = true;
		}
	}

	if (!hit)
	{
		std::wstring err;
		if (!sam().ensure_loaded(err))
		{
			status = err;
			return false;
		}

		// 追跡の起点にできる、近くのキャッシュ済みフレームを探す
		const MaskEntry* base = nullptr;
		int base_frame = 0;
		if (p.track && frame != p.anchor && !inst->masks.empty())
		{
			auto lb = inst->masks.lower_bound(frame);
			if (lb != inst->masks.end() && lb->second.valid && lb->first - frame <= TRACK_SEARCH_FRAMES)
			{
				base = &lb->second;
				base_frame = lb->first;
			}
			if (lb != inst->masks.begin())
			{
				auto pv = std::prev(lb);
				if (pv->second.valid && frame - pv->first <= TRACK_SEARCH_FRAMES)
				{
					if (!base || (frame - pv->first) <= std::abs(base_frame - frame))
					{
						base = &pv->second;
						base_frame = pv->first;
					}
				}
			}
			if (base && (base->w != mw || base->h != mh))
			{
				base = nullptr;
			}
		}

		// シード(物体のおおよその範囲)を決める。起点があればその物体の位置、無ければ指定した範囲
		std::vector<uint8_t> seed;
		std::vector<uint8_t> add_seed, sub_mask;	// 追加した範囲 / 除外した範囲 (囲んだ形で指定した場合)
		BoxF prompt;
		bool from_polygon = false;
		if (base)
		{
			// 起点のマスクを取り出す (古いフレームは圧縮から復元する)
			MaskEntry base_full;
			if (!load_hot(*base, base_full))
			{
				base_full.a.assign((size_t)mw * mh, 0);
			}
			seed.assign(base_full.a.size(), 0);
			for (size_t i = 0; i < seed.size(); i++)
			{
				seed[i] = base_full.a[i] >= 128 ? 1 : 0;
			}
			prompt = expand_box(base->bbox, std::abs(frame - base_frame), w, h);
		}
		else
		{
			if (shapes_have_add(p.shapes))
			{
				from_polygon = true;
				const size_t mn = (size_t)mw * mh;
				add_seed.assign(mn, 0);
				sub_mask.assign(mn, 0);
				std::vector<uint8_t> tmp;
				int minx = w, miny = h, maxx = 0, maxy = 0;
				for (const Shape& sh : p.shapes)
				{
					rasterize_polygon(sh.pts, w, h, mw, mh, tmp);
					std::vector<uint8_t>& dst = sh.subtract ? sub_mask : add_seed;
					for (size_t i = 0; i < mn; i++)
					{
						dst[i] |= tmp[i];
					}
					if (!sh.subtract)
					{
						for (const auto& q : sh.pts)
						{
							minx = std::min(minx, q.first);
							miny = std::min(miny, q.second);
							maxx = std::max(maxx, q.first);
							maxy = std::max(maxy, q.second);
						}
					}
				}
				// 対象 = 追加した範囲から、除外した範囲を引いたもの
				seed.assign(mn, 0);
				for (size_t i = 0; i < mn; i++)
				{
					seed[i] = (add_seed[i] && !sub_mask[i]) ? 1 : 0;
				}
				BoxF b;
				b.x0 = (float)minx;
				b.y0 = (float)miny;
				b.x1 = (float)maxx;
				b.y1 = (float)maxy;
				prompt = expand_box(b, 0, w, h);
			}
			else
			{
				rasterize_rect(p.bx, p.by, p.bw, p.bh, w, h, mw, mh, seed);
				prompt.x0 = (float)std::clamp(p.bx, 0, w - 1);
				prompt.y0 = (float)std::clamp(p.by, 0, h - 1);
				prompt.x1 = (float)std::clamp(p.bx + p.bw, 1, w);
				prompt.y1 = (float)std::clamp(p.by + p.bh, 1, h);
			}
		}

		PromptSet ps;
		ps.box = prompt;
		const float sx = (float)w / mw;
		const float sy = (float)h / mh;
		auto add_points = [&](const std::vector<std::pair<float, float>>& v, float label)
			{
				for (const auto& q : v)
				{
					ps.pts.push_back(q.first * sx);
					ps.pts.push_back(q.second * sy);
					ps.labels.push_back(label);
				}
			};
		bool used_color = false;
		if (!base)
		{
			// 囲んだ範囲(四角形でも同じ)の外側と違う色の部分を「物体上の点」、範囲の縁のすぐ外側や穴の中を「背景上の点」にする
			std::vector<uint8_t> rgb;
			make_small_rgb(buf, w, h, mw, mh, rgb);
			std::vector<std::pair<float, float>> cpos, cneg;
			if (pick_points_by_color(seed, rgb, mw, mh, 4, 4, cpos, cneg))
			{
				add_points(cpos, 1.0f);
				add_points(cneg, 0.0f);
				used_color = true;
			}
		}
		if (!used_color)
		{
			// 物体上の点: シードの内側の、縁から遠い点
			std::vector<std::pair<float, float>> pos;
			pick_points(seed, mw, mh, 3, 6, pos);
			add_points(pos, 1.0f);
			if (from_polygon)
			{
				// 背景上の点: 枠の中で、囲んだ範囲の外側にあり、範囲から遠い点
				std::vector<uint8_t> outside((size_t)mw * mh, 0);
				int x0 = std::clamp((int)(prompt.x0 / sx), 0, mw - 1);
				int x1 = std::clamp((int)(prompt.x1 / sx), 0, mw - 1);
				int y0 = std::clamp((int)(prompt.y0 / sy), 0, mh - 1);
				int y1 = std::clamp((int)(prompt.y1 / sy), 0, mh - 1);
				for (int y = y0; y <= y1; y++)
				{
					for (int x = x0; x <= x1; x++)
					{
						outside[(size_t)y * mw + x] = seed[(size_t)y * mw + x] ? 0 : 1;
					}
				}
				std::vector<std::pair<float, float>> neg;
				pick_points(outside, mw, mh, 2, 12, neg);
				add_points(neg, 0.0f);
			}
		}

		MaskEntry e;
		e.w = mw;
		e.h = mh;
		std::wstring seg_err;
		if (!sam().segment(buf, w, h, ps, mw, mh, e.a, seg_err))
		{
			status = seg_err;
			return false;
		}
		if (from_polygon)
		{
			// 囲んだ範囲の外には出さない (追加した範囲を1画素分だけ広げた内側に制限し、除外した範囲は厳密に除く)
			for (int y = 0; y < mh; y++)
			{
				for (int x = 0; x < mw; x++)
				{
					bool inside = false;
					for (int dy = -1; dy <= 1 && !inside; dy++)
					{
						for (int dx = -1; dx <= 1; dx++)
						{
							int xx = x + dx;
							int yy = y + dy;
							if (xx >= 0 && xx < mw && yy >= 0 && yy < mh && add_seed[(size_t)yy * mw + xx])
							{
								inside = true;
								break;
							}
						}
					}
					// 追加した範囲の外と、除外した範囲の中は、対象にしない
					if (!inside || sub_mask[(size_t)y * mw + x])
					{
						e.a[(size_t)y * mw + x] = 0;
					}
				}
			}
		}
		memcpy(e.sig, cur_sig, sizeof(e.sig));
		e.thumb = cur_thumb;
		e.tw = tw;
		e.th = th;
		e.age = 0;

		// 物体の外接枠を求める
		int minx = mw, miny = mh, maxx = -1, maxy = -1;
		int count = 0;
		for (int y = 0; y < mh; y++)
		{
			for (int x = 0; x < mw; x++)
			{
				if (e.a[(size_t)y * mw + x] >= 128)
				{
					minx = std::min(minx, x);
					miny = std::min(miny, y);
					maxx = std::max(maxx, x);
					maxy = std::max(maxy, y);
					count++;
				}
			}
		}
		if (count >= 8)
		{
			e.valid = true;
			e.bbox.x0 = (float)minx * w / mw;
			e.bbox.y0 = (float)miny * h / mh;
			e.bbox.x1 = (float)(maxx + 1) * w / mw;
			e.bbox.y1 = (float)(maxy + 1) * h / mh;
		}
		else
		{
			e.valid = false;
			e.bbox = prompt;
		}
		e.chash = chash;
		cur = e;
		store_entry(inst, frame, std::move(e));
	}

	has_prev = false;
	auto pit = inst->masks.find(frame - 1);
	if (pit != inst->masks.end() && pit->second.valid && load_hot(pit->second, prev))
	{
		has_prev = true;
	}
	return true;
}

//---------------------------------------------------------------------
//	フィルタ設定項目定義
//---------------------------------------------------------------------
static void on_clear_cache(EDIT_SECTION*, OBJECT_HANDLE, LPCWSTR, LPCWSTR)
{
	g_generation++;
	set_status(L"解析キャッシュを破棄しました");
}

auto g_group_box = FILTER_ITEM_GROUP(L"対象の枠");
auto g_box_x = FILTER_ITEM_TRACK(ITEM_BOX_X, 0, 0, 16384, 1);
auto g_box_y = FILTER_ITEM_TRACK(ITEM_BOX_Y, 0, 0, 16384, 1);
auto g_box_w = FILTER_ITEM_TRACK(ITEM_BOX_W, 0, 0, 16384, 1);
auto g_box_h = FILTER_ITEM_TRACK(ITEM_BOX_H, 0, 0, 16384, 1);
auto g_anchor = FILTER_ITEM_TRACK(ITEM_ANCHOR, 0, 0, 1000000, 1);
auto g_shape = FILTER_ITEM_STRING(ITEM_SHAPE, L"");
auto g_group_opt = FILTER_ITEM_GROUP(L"動作");
auto g_track = FILTER_ITEM_CHECK(L"追跡", true);
auto g_interval = FILTER_ITEM_TRACK(L"解析間隔(フレーム)", 4, 1, 30, 1);
auto g_invert = FILTER_ITEM_CHECK(L"反転", false);
auto g_group_edge = FILTER_ITEM_GROUP(L"輪郭");
auto g_feather = FILTER_ITEM_TRACK(L"ぼかし", 1.0, 0.0, 100.0, 0.1);
auto g_grow = FILTER_ITEM_TRACK(L"拡大縮小", 0.0, -50.0, 50.0, 0.1);
auto g_smooth = FILTER_ITEM_TRACK(L"時間平滑化", 0.3, 0.0, 0.9, 0.01);
auto g_threshold = FILTER_ITEM_TRACK(L"しきい値", 0.5, 0.05, 0.95, 0.01);
auto g_refine = FILTER_ITEM_TRACK(L"輪郭補正", 6.0, 0.0, 30.0, 1.0);
auto g_matte = FILTER_ITEM_TRACK(L"縁の精密化", 2.0, 0.0, 10.0, 1.0);
auto g_despeckle = FILTER_ITEM_TRACK(L"ゴミ除去", 6.0, 0.0, 30.0, 1.0);
auto g_smooth_shape = FILTER_ITEM_TRACK(L"形のなめらか化", 1.0, 0.0, 10.0, 1.0);
auto g_alpha_low = FILTER_ITEM_TRACK(L"透明度の下限", 0.10, 0.0, 0.95, 0.01);
auto g_alpha_high = FILTER_ITEM_TRACK(L"透明度の上限", 0.90, 0.05, 1.0, 0.01);
auto g_unify = FILTER_ITEM_TRACK(L"色の均一化", 0.0, 0.0, 10.0, 1.0);
auto g_edge_color = FILTER_ITEM_TRACK(L"縁の色補正", 2.0, 0.0, 10.0, 1.0);
auto g_clear = FILTER_ITEM_BUTTON(L"解析キャッシュを破棄", on_clear_cache);

void* g_items[] = {
	&g_group_box, &g_box_x, &g_box_y, &g_box_w, &g_box_h, &g_anchor, &g_shape,
	&g_group_opt, &g_track, &g_interval, &g_invert,
	&g_group_edge, &g_feather, &g_grow, &g_smooth, &g_threshold, &g_refine, &g_matte, &g_edge_color, &g_despeckle, &g_smooth_shape, &g_alpha_low, &g_alpha_high, &g_unify,
	&g_clear,
	nullptr };

static Params read_params(int w, int h)
{
	Params p;
	parse_shapes(g_shape.value, w, h, p.shapes);
	p.bx = (int)std::lround(g_box_x.value);
	p.by = (int)std::lround(g_box_y.value);
	p.bw = (int)std::lround(g_box_w.value);
	p.bh = (int)std::lround(g_box_h.value);
	p.anchor = (int)std::lround(g_anchor.value);
	p.track = g_track.value;
	p.interval = std::clamp((int)std::lround(g_interval.value), 1, 30);
	p.invert = g_invert.value;
	p.feather = (float)g_feather.value;
	p.grow = (float)g_grow.value;
	p.smooth = (float)g_smooth.value;
	p.threshold = (float)g_threshold.value;
	p.refine = (float)g_refine.value;
	p.edge_color = (float)g_edge_color.value;
	p.matte = (float)g_matte.value;
	p.despeckle = (float)g_despeckle.value;
	p.smooth_shape = (float)g_smooth_shape.value;
	p.alpha_low = (float)g_alpha_low.value;
	p.alpha_high = (float)g_alpha_high.value;
	p.unify = (float)g_unify.value;
	return p;
}

//---------------------------------------------------------------------
//	画像フィルタ処理
//---------------------------------------------------------------------
static uint64_t result_key(uint64_t chash, const Params& p, int w, int h)
{
	uint64_t hash = params_hash(p, w, h) ^ (chash * 0x9E3779B97F4A7C15ull);
	auto mix = [&hash](int v)
		{
			hash ^= (uint64_t)(uint32_t)v;
			hash *= 1099511628211ull;
		};
	mix(p.invert ? 1 : 0);
	mix((int)std::lround(p.feather * 100.0f));
	mix((int)std::lround(p.grow * 100.0f));
	mix((int)std::lround(p.smooth * 100.0f));
	mix((int)std::lround(p.threshold * 100.0f));
	mix((int)std::lround(p.refine * 100.0f));
	mix((int)std::lround(p.edge_color * 100.0f));
	mix((int)std::lround(p.matte * 100.0f));
	mix((int)std::lround(p.despeckle * 100.0f));
	mix((int)std::lround(p.smooth_shape * 100.0f));
	mix((int)std::lround(p.alpha_low * 1000.0f));
	mix((int)std::lround(p.alpha_high * 1000.0f));
	mix((int)std::lround(p.unify * 100.0f));
	mix(g_generation.load());
	return hash;
}

static bool func_proc_video(FILTER_PROC_VIDEO* video)
{
	const int w = video->object->width;
	const int h = video->object->height;
	if (w <= 0 || h <= 0)
	{
		return true;
	}
	const int frame = video->object->frame;

	thread_local std::vector<PIXEL_RGBA> buf;
	buf.resize((size_t)w * h);
	video->get_image_data(buf.data());

	Params p = read_params(w, h);

	// 枠選択ウィンドウには、オブジェクト設定で選択中のオブジェクトの画像だけを渡す
	// 再生・出力中 (フレームが連続して呼ばれる間) は、渡す頻度を減らして負荷を抑える
	ULONGLONG now = GetTickCount64();
	ULONGLONG last_call = g_last_call.exchange(now);
	bool continuous = (now - last_call) < 40;
	bool show_preview = g_preview_visible.load();
	if (show_preview && continuous && now - g_last_publish.load() < 120)
	{
		show_preview = false;
	}
	if (show_preview && video->edit && video->edit->get_focus_object)
	{
		OBJECT_HANDLE focus = video->edit->get_focus_object();
		show_preview = (!focus || focus == video->object->handle);
	}
	if (show_preview)
	{
		g_last_publish.store(now);
	}

	bool has_seed = shapes_have_add(p.shapes) || (p.bw >= 4 && p.bh >= 4);
	if (!has_seed)
	{
		if (show_preview)
		{
			publish_preview(buf.data(), w, h, frame, p, nullptr, 0, 0, nullptr, L"独自ウィンドウで、対象の物体を囲むようにドラッグしてください (Shift = 四角形、右ドラッグ = 除外)");
		}
		return true;
	}

	// 直前と同じ入力 (静止画など) なら、前回の結果をそのまま返す
	const uint64_t chash = content_hash(buf.data(), w, h);
	const bool same_as_prev = (g_prev_chash.exchange(chash) == chash);
	const uint64_t rkey = result_key(chash, p, w, h);
	{
		std::lock_guard<std::mutex> lock(g_result_mtx);
		if (g_last_result.valid && g_last_result.key == rkey && g_last_result.w == w && g_last_result.h == h)
		{
			if (show_preview)
			{
				publish_preview(buf.data(), w, h, frame, p, &g_last_result.m, g_last_result.mw, g_last_result.mh,
					g_last_result.has_bbox ? &g_last_result.bbox : nullptr, g_last_result.msg);
			}
			video->set_image_data(g_last_result.img.data(), w, h);
			return true;
		}
	}

	MaskEntry cur, prev;
	bool has_prev = false;
	bool content_hit = false;
	std::wstring status;
	bool ok = false;
	try
	{
		std::lock_guard<std::mutex> lock(g_cache_mtx);
		CacheBucket* bucket = acquire_bucket(p, w, h);
		ok = get_mask(bucket, p, frame, buf.data(), w, h, chash, cur, prev, has_prev, content_hit, status);
	}
	catch (const std::exception& e)
	{
		status = L"処理中にエラーが発生しました: " + widen(e.what());
		ok = false;
	}
	if (!ok)
	{
		if (show_preview)
		{
			publish_preview(buf.data(), w, h, frame, p, nullptr, 0, 0, nullptr, status);
		}
		return true;
	}

	float scale = (float)cur.w / (float)w;
	std::vector<float> m;
	build_float_mask(cur, has_prev ? &prev : nullptr, p, scale, m, false);
	std::vector<float> m_preview;	// プレビュー用 (拡大縮小も大まかに反映する)
	if (show_preview)
	{
		build_float_mask(cur, has_prev ? &prev : nullptr, p, scale, m_preview, true);
	}

	std::wstring msg = cur.valid ? std::wstring(L"検出しました (") + sam().device_name() + L")" : L"物体を検出できませんでした。枠を描き直してください";
	if (show_preview)
	{
		// 処理前の画像に、対象外を暗くして重ねたものを渡す
		publish_preview(buf.data(), w, h, frame, p, &m_preview, cur.w, cur.h, cur.valid ? &cur.bbox : nullptr, msg);
	}

	apply_mask_refined(buf.data(), w, h, m, cur.w, cur.h, p.invert, p.refine, p.feather, p.grow, p.edge_color, p.matte, p.smooth_shape, p.alpha_low, p.alpha_high, p.unify);
	video->set_image_data(buf.data(), w, h);

	// 同じ内容のフレームが繰り返されている (静止画など) ときは、結果を覚えておく
	if (content_hit || same_as_prev)
	{
		std::lock_guard<std::mutex> lock(g_result_mtx);
		g_last_result.key = rkey;
		g_last_result.w = w;
		g_last_result.h = h;
		g_last_result.img.assign(buf.begin(), buf.end());
		g_last_result.m = m_preview.empty() ? std::move(m) : std::move(m_preview);
		g_last_result.mw = cur.w;
		g_last_result.mh = cur.h;
		g_last_result.bbox = cur.bbox;
		g_last_result.has_bbox = cur.valid;
		g_last_result.msg = msg;
		g_last_result.valid = true;
	}
	return true;
}

FILTER_PLUGIN_TABLE g_filter_table = {
	FILTER_PLUGIN_TABLE::FLAG_VIDEO | FILTER_PLUGIN_TABLE::FLAG_FILTER,
	FILTER_NAME,
	L"クリッピング",
	L"AI Object Mask version 0.03",
	g_items,
	func_proc_video,
	nullptr,
	nullptr,
	nullptr
};

//---------------------------------------------------------------------
//	枠選択ウィンドウ
//---------------------------------------------------------------------
#define WINDOW_CLASS_NAME	L"AiObjectMaskWindow"
#define WINDOW_TITLE		L"AIオブジェクトマスク"
#define IDC_CLEAR_BOX		2001
#define IDC_CLEAR_CACHE		2002
#define IDC_UNDO			2003
#define IDC_MODE			2004
#define IDT_REFRESH			1

struct WindowState
{
	bool dragging = false;
	bool drag_right = false;		// 右ボタンでのドラッグ
	bool subtract = false;			// このドラッグは「除外」
	bool subtract_mode = false;		// ボタンで切り替える「除外モード」 (左ドラッグも除外になる)
	HWND mode_button = nullptr;
	bool rect_mode = false;			// Shiftを押しながらのドラッグ = 四角形
	std::vector<POINT> path;		// 自由線の軌跡 (ウィンドウ座標)
	POINT p0 = {}, p1 = {};
	RECT image_rect = {};
	uint64_t shown_version = ~0ull;
	HFONT font = nullptr;
};

static WindowState g_win;

static int top_bar_height()
{
	int item = g_config ? g_config->get_layout_size(g_config, "SettingItemHeight") : 24;
	if (item <= 0)
	{
		item = 24;
	}
	return item + 40;
}

struct BoxRequest
{
	int x = 0, y = 0, w = 0, h = 0;
	int frame = 0;
	std::string shape;				// 範囲の座標 ("+x,y;x,y;|-x,y;..."。+は追加、-は除外)
	std::wstring note;				// 設定後に表示する文
};

static BoxRequest g_request;

// 編集セクション内で、選択中のオブジェクトのフィルタに枠を書き込む
static void write_box(void* param, EDIT_SECTION* edit)
{
	BoxRequest* r = (BoxRequest*)param;
	OBJECT_HANDLE obj = edit->get_focus_object();
	if (!obj || edit->count_object_effect(obj, FILTER_NAME) == 0)
	{
		set_status(L"範囲を設定できません。AIオブジェクトマスクを付けたオブジェクトを選択してください");
		return;
	}
	char v[32];
	snprintf(v, sizeof(v), "%d", r->x);
	edit->set_object_item_value(obj, FILTER_NAME, ITEM_BOX_X, v);
	snprintf(v, sizeof(v), "%d", r->y);
	edit->set_object_item_value(obj, FILTER_NAME, ITEM_BOX_Y, v);
	snprintf(v, sizeof(v), "%d", r->w);
	edit->set_object_item_value(obj, FILTER_NAME, ITEM_BOX_W, v);
	snprintf(v, sizeof(v), "%d", r->h);
	edit->set_object_item_value(obj, FILTER_NAME, ITEM_BOX_H, v);
	snprintf(v, sizeof(v), "%d", r->frame);
	edit->set_object_item_value(obj, FILTER_NAME, ITEM_ANCHOR, v);
	edit->set_object_item_value(obj, FILTER_NAME, ITEM_SHAPE, r->shape.c_str());
	if (!r->note.empty())
	{
		set_status(r->note);
	}
	else
	{
		set_status(r->w > 0 ? L"範囲を設定しました。解析します..." : L"範囲をクリアしました");
	}
}

static void commit_box(int x, int y, int w, int h, int frame, const std::string& shape, const std::wstring& note = L"")
{
	g_request.x = x;
	g_request.y = y;
	g_request.w = w;
	g_request.h = h;
	g_request.frame = frame;
	g_request.shape = shape;
	g_request.note = note;
	if (g_edit_handle)
	{
		g_edit_handle->call_edit_section_param(&g_request, write_box);
	}
}

// 範囲の一覧を、選択中のオブジェクトの設定へ書き込む (枠は、追加した範囲全体を囲む四角形にする)
static void commit_shapes(const ShapeList& list, int anchor, const std::wstring& note)
{
	int minx = INT_MAX, miny = INT_MAX, maxx = 0, maxy = 0;
	for (const Shape& sh : list)
	{
		if (sh.subtract)
		{
			continue;
		}
		for (const auto& q : sh.pts)
		{
			minx = std::min(minx, q.first);
			miny = std::min(miny, q.second);
			maxx = std::max(maxx, q.first);
			maxy = std::max(maxy, q.second);
		}
	}
	{
		// 続けて操作しても、直前の内容が失われないよう、すぐに共有の状態へ反映する
		std::lock_guard<std::mutex> lock(g_preview.mtx);
		g_preview.shapes = list;
		g_preview.anchor = anchor;
	}
	if (minx == INT_MAX)
	{
		commit_box(0, 0, 0, 0, 0, "", note);
		return;
	}
	commit_box(minx, miny, maxx - minx, maxy - miny, anchor, shapes_to_string(list), note);
}

// 新しい範囲 (追加または除外) を、今ある範囲の一覧に加える
static void add_shape(const PolyPoints& pts, bool subtract, int frame)
{
	ShapeList list;
	int anchor;
	{
		std::lock_guard<std::mutex> lock(g_preview.mtx);
		list = g_preview.shapes;
		anchor = g_preview.anchor;
	}
	const bool had_add = shapes_have_add(list);
	if (subtract && !had_add)
	{
		set_status(L"除外は、追加した範囲がある状態で行ってください");
		return;
	}
	if (list.size() >= 16)
	{
		set_status(L"範囲は16個までです。範囲をクリアするか、直前を取り消してください");
		return;
	}
	std::wstring note = subtract ? L"除外する範囲を追加しました。解析します..." : L"範囲を追加しました。解析します...";
	if (!had_add)
	{
		anchor = frame;		// 最初に追加した範囲のフレームが、基準フレームになる
	}
	else if (frame != anchor)
	{
		note = L"基準フレームと違うフレームで指定しました。動く映像では、位置がずれる場合があります";
	}
	Shape sh;
	sh.subtract = subtract;
	sh.pts = pts;
	list.push_back(std::move(sh));
	commit_shapes(list, anchor, note);
}

// 直前に指定した範囲を取り消す
static void undo_shape()
{
	ShapeList list;
	int anchor;
	{
		std::lock_guard<std::mutex> lock(g_preview.mtx);
		list = g_preview.shapes;
		anchor = g_preview.anchor;
	}
	if (list.empty())
	{
		set_status(L"取り消す範囲がありません");
		return;
	}
	list.pop_back();
	if (!shapes_have_add(list))
	{
		list.clear();	// 追加した範囲が無くなったら、除外だけが残らないよう全部消す
	}
	commit_shapes(list, anchor, list.empty() ? std::wstring(L"範囲をクリアしました") : std::wstring(L"直前の範囲を取り消しました"));
}

// ウィンドウ上の座標 → 元画像の座標
static bool window_to_source(int mx, int my, int src_w, int src_h, int& ix, int& iy)
{
	const RECT& r = g_win.image_rect;
	int rw = r.right - r.left;
	int rh = r.bottom - r.top;
	if (rw <= 0 || rh <= 0 || src_w <= 0 || src_h <= 0)
	{
		return false;
	}
	ix = std::clamp((int)((int64_t)(mx - r.left) * src_w / rw), 0, src_w);
	iy = std::clamp((int)((int64_t)(my - r.top) * src_h / rh), 0, src_h);
	return true;
}

static void draw_rect_outline(HDC dc, int l, int t, int r, int b, COLORREF color, int width)
{
	HPEN pen = CreatePen(PS_SOLID, width, color);
	HGDIOBJ old_pen = SelectObject(dc, pen);
	HGDIOBJ old_brush = SelectObject(dc, GetStockObject(NULL_BRUSH));
	Rectangle(dc, l, t, r, b);
	SelectObject(dc, old_brush);
	SelectObject(dc, old_pen);
	DeleteObject(pen);
}

static void paint_window(HWND hwnd)
{
	PAINTSTRUCT ps;
	HDC hdc = BeginPaint(hwnd, &ps);
	RECT rc;
	GetClientRect(hwnd, &rc);
	int cw = rc.right - rc.left;
	int ch = rc.bottom - rc.top;
	HDC dc = CreateCompatibleDC(hdc);
	HBITMAP bmp = CreateCompatibleBitmap(hdc, std::max(1, cw), std::max(1, ch));
	HGDIOBJ old_bmp = SelectObject(dc, bmp);

	HBRUSH bg = CreateSolidBrush(RGB(32, 32, 32));
	FillRect(dc, &rc, bg);
	DeleteObject(bg);

	std::vector<uint32_t> bgra;
	int pw, ph, src_w, src_h, bx, by, bw, bh;
	bool has_bbox;
	BoxF bbox;
	ShapeList shapes;
	std::wstring status;
	{
		std::lock_guard<std::mutex> lock(g_preview.mtx);
		bgra = g_preview.bgra;
		pw = g_preview.pw;
		ph = g_preview.ph;
		src_w = g_preview.src_w;
		src_h = g_preview.src_h;
		bx = g_preview.bx;
		by = g_preview.by;
		bw = g_preview.bw;
		bh = g_preview.bh;
		has_bbox = g_preview.has_bbox;
		bbox = g_preview.bbox;
		shapes = g_preview.shapes;
		status = g_preview.status;
		g_win.shown_version = g_preview.version;
	}

	int top = top_bar_height();
	SetBkMode(dc, TRANSPARENT);
	SetTextColor(dc, RGB(230, 230, 230));
	if (g_win.font)
	{
		SelectObject(dc, g_win.font);
	}
	std::wstring line = status.empty() ? std::wstring(L"AIオブジェクトマスクを付けたオブジェクトを選択して、再生バーを動かしてください") : status;
	RECT text_rc = { 8, top - 30, cw - 8, top - 4 };
	DrawTextW(dc, line.c_str(), -1, &text_rc, DT_LEFT | DT_VCENTER | DT_SINGLELINE | DT_END_ELLIPSIS);

	g_win.image_rect = { 0, 0, 0, 0 };
	if (!bgra.empty() && pw > 0 && ph > 0)
	{
		int area_w = cw;
		int area_h = std::max(1, ch - top);
		int dw = area_w;
		int dh = (int)((int64_t)ph * dw / pw);
		if (dh > area_h)
		{
			dh = area_h;
			dw = (int)((int64_t)pw * dh / ph);
		}
		int dx = (area_w - dw) / 2;
		int dy = top + (area_h - dh) / 2;
		g_win.image_rect = { dx, dy, dx + dw, dy + dh };

		BITMAPINFO bi = {};
		bi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
		bi.bmiHeader.biWidth = pw;
		bi.bmiHeader.biHeight = -ph;
		bi.bmiHeader.biPlanes = 1;
		bi.bmiHeader.biBitCount = 32;
		bi.bmiHeader.biCompression = BI_RGB;
		SetStretchBltMode(dc, HALFTONE);
		StretchDIBits(dc, dx, dy, dw, dh, 0, 0, pw, ph, bgra.data(), &bi, DIB_RGB_COLORS, SRCCOPY);

		auto to_screen = [&](float x, float y, int& sx, int& sy)
			{
				sx = dx + (int)(x * dw / src_w);
				sy = dy + (int)(y * dh / src_h);
			};
		if (!shapes.empty())
		{
			for (const Shape& sh : shapes)
			{
				if (sh.pts.size() < 3)
				{
					continue;
				}
				std::vector<POINT> pts(sh.pts.size());
				for (size_t i = 0; i < sh.pts.size(); i++)
				{
					int sx, sy;
					to_screen((float)sh.pts[i].first, (float)sh.pts[i].second, sx, sy);
					pts[i] = { sx, sy };
				}
				// 追加は黄色、除外は水色で描く
				HPEN pen = CreatePen(PS_SOLID, 2, sh.subtract ? RGB(0, 200, 255) : RGB(255, 210, 0));
				HGDIOBJ old_pen = SelectObject(dc, pen);
				HGDIOBJ old_brush = SelectObject(dc, GetStockObject(NULL_BRUSH));
				Polygon(dc, pts.data(), (int)pts.size());
				SelectObject(dc, old_brush);
				SelectObject(dc, old_pen);
				DeleteObject(pen);
			}
		}
		else if (bw > 0 && bh > 0)
		{
			int l, t, r, b;
			to_screen((float)bx, (float)by, l, t);
			to_screen((float)(bx + bw), (float)(by + bh), r, b);
			draw_rect_outline(dc, l, t, r, b, RGB(255, 210, 0), 2);
		}
		if (has_bbox)
		{
			int l, t, r, b;
			to_screen(bbox.x0, bbox.y0, l, t);
			to_screen(bbox.x1, bbox.y1, r, b);
			draw_rect_outline(dc, l, t, r, b, RGB(0, 220, 120), 1);
		}
	}
	if (g_win.dragging)
	{
		if (g_win.rect_mode)
		{
			int l = std::min(g_win.p0.x, g_win.p1.x);
			int t = std::min(g_win.p0.y, g_win.p1.y);
			int r = std::max(g_win.p0.x, g_win.p1.x);
			int b = std::max(g_win.p0.y, g_win.p1.y);
			draw_rect_outline(dc, l, t, r, b, g_win.subtract ? RGB(0, 200, 255) : RGB(255, 80, 80), 2);
		}
		else if (g_win.path.size() >= 2)
		{
			// 軌跡と、始点へ戻る線(閉じたときの形)を描く
			COLORREF drag_color = g_win.subtract ? RGB(0, 200, 255) : RGB(255, 80, 80);
			HPEN pen = CreatePen(PS_SOLID, 2, drag_color);
			HGDIOBJ old_pen = SelectObject(dc, pen);
			Polyline(dc, g_win.path.data(), (int)g_win.path.size());
			DeleteObject(SelectObject(dc, old_pen));
			HPEN thin = CreatePen(PS_DOT, 1, drag_color);
			old_pen = SelectObject(dc, thin);
			MoveToEx(dc, g_win.path.back().x, g_win.path.back().y, nullptr);
			LineTo(dc, g_win.path.front().x, g_win.path.front().y);
			DeleteObject(SelectObject(dc, old_pen));
		}
	}

	BitBlt(hdc, 0, 0, cw, ch, dc, 0, 0, SRCCOPY);
	SelectObject(dc, old_bmp);
	DeleteObject(bmp);
	DeleteDC(dc);
	EndPaint(hwnd, &ps);
}

static LRESULT CALLBACK window_proc(HWND hwnd, UINT message, WPARAM wparam, LPARAM lparam)
{
	switch (message)
	{
	case WM_CREATE:
		SetTimer(hwnd, IDT_REFRESH, 100, nullptr);
		return 0;

	case WM_DESTROY:
		KillTimer(hwnd, IDT_REFRESH);
		return 0;

	case WM_TIMER:
		if (wparam == IDT_REFRESH)
		{
			g_preview_visible.store(IsWindowVisible(hwnd) != FALSE);
			uint64_t v;
			{
				std::lock_guard<std::mutex> lock(g_preview.mtx);
				v = g_preview.version;
			}
			if (v != g_win.shown_version && !g_win.dragging)
			{
				InvalidateRect(hwnd, nullptr, FALSE);
			}
		}
		return 0;

	case WM_ERASEBKGND:
		return 1;

	case WM_PAINT:
		paint_window(hwnd);
		return 0;

	case WM_SIZE:
		InvalidateRect(hwnd, nullptr, FALSE);
		return 0;

	case WM_LBUTTONDOWN:
	case WM_RBUTTONDOWN:
	{
		int mx = GET_X_LPARAM(lparam);
		int my = GET_Y_LPARAM(lparam);
		const RECT& r = g_win.image_rect;
		if (!g_win.dragging && mx >= r.left && mx < r.right && my >= r.top && my < r.bottom)
		{
			g_win.dragging = true;
			g_win.drag_right = (message == WM_RBUTTONDOWN);
			// 右ボタン、または除外モードのときの左ボタンは「除外」
			g_win.subtract = g_win.drag_right || g_win.subtract_mode;
			g_win.rect_mode = (GetKeyState(VK_SHIFT) & 0x8000) != 0;
			g_win.p0 = { mx, my };
			g_win.p1 = { mx, my };
			g_win.path.clear();
			g_win.path.push_back(g_win.p0);
			SetCapture(hwnd);
		}
		return 0;
	}

	case WM_MOUSEMOVE:
		if (g_win.dragging)
		{
			const RECT& r = g_win.image_rect;
			g_win.p1.x = std::clamp((int)GET_X_LPARAM(lparam), (int)r.left, (int)r.right);
			g_win.p1.y = std::clamp((int)GET_Y_LPARAM(lparam), (int)r.top, (int)r.bottom);
			if (!g_win.rect_mode)
			{
				const POINT& last = g_win.path.back();
				int dx = g_win.p1.x - last.x;
				int dy = g_win.p1.y - last.y;
				if (dx * dx + dy * dy >= 4)
				{
					g_win.path.push_back(g_win.p1);
				}
			}
			InvalidateRect(hwnd, nullptr, FALSE);
		}
		return 0;

	case WM_LBUTTONUP:
	case WM_RBUTTONUP:
		if (g_win.dragging && ((message == WM_RBUTTONUP) == g_win.drag_right))
		{
			g_win.dragging = false;
			ReleaseCapture();
			int src_w, src_h, frame;
			{
				std::lock_guard<std::mutex> lock(g_preview.mtx);
				src_w = g_preview.src_w;
				src_h = g_preview.src_h;
				frame = g_preview.frame;
			}
			PolyPoints pts;
			if (g_win.rect_mode)
			{
				// 四角形は、4つの角の多角形にする
				int x0, y0, x1, y1;
				if (window_to_source(std::min(g_win.p0.x, g_win.p1.x), std::min(g_win.p0.y, g_win.p1.y), src_w, src_h, x0, y0)
					&& window_to_source(std::max(g_win.p0.x, g_win.p1.x), std::max(g_win.p0.y, g_win.p1.y), src_w, src_h, x1, y1)
					&& x1 - x0 >= 4 && y1 - y0 >= 4)
				{
					pts = { { x0, y0 }, { x1, y0 }, { x1, y1 }, { x0, y1 } };
				}
			}
			else if (g_win.path.size() >= 8)
			{
				// 軌跡を元画像の座標にして、近すぎる点を間引く
				int min_step = std::max(2, src_w / 240);
				for (const POINT& q : g_win.path)
				{
					int ix, iy;
					if (!window_to_source(q.x, q.y, src_w, src_h, ix, iy))
					{
						continue;
					}
					if (pts.empty())
					{
						pts.push_back({ ix, iy });
						continue;
					}
					int dx = ix - pts.back().first;
					int dy = iy - pts.back().second;
					if (dx * dx + dy * dy >= min_step * min_step)
					{
						pts.push_back({ ix, iy });
					}
				}
				// 点が多すぎる場合は等間隔に減らす (設定値の文字列を短く保つ)
				const size_t max_points = 48;
				if (pts.size() > max_points)
				{
					PolyPoints reduced;
					for (size_t i = 0; i < max_points; i++)
					{
						reduced.push_back(pts[i * pts.size() / max_points]);
					}
					pts.swap(reduced);
				}
				int minx = src_w, miny = src_h, maxx = 0, maxy = 0;
				for (const auto& q : pts)
				{
					minx = std::min(minx, q.first);
					miny = std::min(miny, q.second);
					maxx = std::max(maxx, q.first);
					maxy = std::max(maxy, q.second);
				}
				if (pts.size() < 3 || maxx - minx < 4 || maxy - miny < 4)
				{
					pts.clear();
				}
			}
			if (pts.size() >= 3)
			{
				add_shape(pts, g_win.subtract, frame);
			}
			g_win.path.clear();
			InvalidateRect(hwnd, nullptr, FALSE);
		}
		return 0;

	case WM_COMMAND:
		switch (LOWORD(wparam))
		{
		case IDC_CLEAR_BOX:
			commit_shapes(ShapeList(), 0, L"範囲をクリアしました");
			SetFocus(nullptr);
			return 0;

		case IDC_UNDO:
			undo_shape();
			SetFocus(nullptr);
			return 0;

		case IDC_MODE:
			g_win.subtract_mode = !g_win.subtract_mode;
			SetWindowTextW(g_win.mode_button, g_win.subtract_mode ? L"モード: 除外" : L"モード: 追加");
			SetFocus(nullptr);
			return 0;

		case IDC_CLEAR_CACHE:
			g_generation++;
			set_status(L"解析キャッシュを破棄しました");
			SetFocus(nullptr);
			return 0;
		}
		break;
	}
	return DefWindowProc(hwnd, message, wparam, lparam);
}

//---------------------------------------------------------------------
//	汎用プラグイン構造体定義
//---------------------------------------------------------------------
COMMON_PLUGIN_TABLE g_common_plugin_table = {
	L"AI Object Mask",
	L"AI Object Mask version 0.01",
};

EXTERN_C __declspec(dllexport) DWORD RequiredVersion()
{
	return 2003300;
}

EXTERN_C __declspec(dllexport) void InitializeLogger(LOG_HANDLE* handle)
{
	g_logger = handle;
}

EXTERN_C __declspec(dllexport) void InitializeConfig(CONFIG_HANDLE* handle)
{
	g_config = handle;
}

EXTERN_C __declspec(dllexport) bool InitializePlugin(DWORD version)
{
	g_flush_stop.store(false);
	g_flush_thread = std::thread(flush_loop);
	return true;
}

EXTERN_C __declspec(dllexport) void UninitializePlugin()
{
	g_flush_stop.store(true);
	if (g_flush_thread.joinable())
	{
		g_flush_thread.join();
	}
	delete g_sam;
	g_sam = nullptr;
}

EXTERN_C __declspec(dllexport) COMMON_PLUGIN_TABLE* GetCommonPluginTable(void)
{
	return &g_common_plugin_table;
}

//---------------------------------------------------------------------
//	プラグイン登録関数
//---------------------------------------------------------------------
EXTERN_C __declspec(dllexport) void RegisterPlugin(HOST_APP_TABLE* host)
{
	// フィルタ効果を登録
	host->register_filter_plugin(&g_filter_table);

	// 枠選択ウィンドウを作成
	WNDCLASSEXW wcex = {};
	wcex.cbSize = sizeof(WNDCLASSEXW);
	wcex.lpszClassName = WINDOW_CLASS_NAME;
	wcex.lpfnWndProc = window_proc;
	wcex.hInstance = GetModuleHandle(0);
	wcex.hbrBackground = nullptr;
	wcex.hCursor = LoadCursor(NULL, IDC_CROSS);
	if (!RegisterClassExW(&wcex))
	{
		return;
	}
	HWND hwnd = CreateWindowExW(
		0,
		WINDOW_CLASS_NAME,
		WINDOW_TITLE,
		WS_POPUP, // 親ウィンドウの指定無しでWS_CHILDが作れないので一旦WS_POPUPで作成する
		CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT,
		nullptr,
		nullptr,
		GetModuleHandle(0),
		nullptr);
	if (!hwnd)
	{
		return;
	}

	g_win.font = (HFONT)GetStockObject(DEFAULT_GUI_FONT);
	int item_h = g_config ? g_config->get_layout_size(g_config, "SettingItemHeight") : 24;
	if (item_h <= 0)
	{
		item_h = 24;
	}
	HWND b1 = CreateWindowExW(0, L"BUTTON", L"範囲をクリア", WS_VISIBLE | WS_CHILD | BS_PUSHBUTTON,
		8, 4, 100, item_h, hwnd, (HMENU)IDC_CLEAR_BOX, GetModuleHandle(0), nullptr);
	HWND b3 = CreateWindowExW(0, L"BUTTON", L"直前を取り消し", WS_VISIBLE | WS_CHILD | BS_PUSHBUTTON,
		116, 4, 120, item_h, hwnd, (HMENU)IDC_UNDO, GetModuleHandle(0), nullptr);
	HWND b4 = CreateWindowExW(0, L"BUTTON", L"モード: 追加", WS_VISIBLE | WS_CHILD | BS_PUSHBUTTON,
		244, 4, 120, item_h, hwnd, (HMENU)IDC_MODE, GetModuleHandle(0), nullptr);
	HWND b2 = CreateWindowExW(0, L"BUTTON", L"解析キャッシュを破棄", WS_VISIBLE | WS_CHILD | BS_PUSHBUTTON,
		372, 4, 170, item_h, hwnd, (HMENU)IDC_CLEAR_CACHE, GetModuleHandle(0), nullptr);
	g_win.mode_button = b4;
	SendMessage(b1, WM_SETFONT, (WPARAM)g_win.font, TRUE);
	SendMessage(b3, WM_SETFONT, (WPARAM)g_win.font, TRUE);
	SendMessage(b4, WM_SETFONT, (WPARAM)g_win.font, TRUE);
	SendMessage(b2, WM_SETFONT, (WPARAM)g_win.font, TRUE);

	host->register_window_client(WINDOW_TITLE, hwnd);
	g_edit_handle = host->create_edit_handle();
}