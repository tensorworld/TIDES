#pragma once

#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

// Minimal zero-dependency RFC 8259 header-only JSON parser (C++17).
namespace bsptc {
namespace mini_json {

struct Value;
using Array = std::vector<Value>;
using Object = std::map<std::string, Value>;

struct Value {
    using Storage = std::variant<std::nullptr_t, bool, std::int64_t, double, std::string, Array, Object>;
    Storage data{nullptr};

    Value() = default;
    explicit Value(std::nullptr_t) : data(nullptr) {}
    explicit Value(bool v) : data(v) {}
    explicit Value(std::int64_t v) : data(v) {}
    explicit Value(double v) : data(v) {}
    explicit Value(std::string v) : data(std::move(v)) {}
    explicit Value(Array v) : data(std::move(v)) {}
    explicit Value(Object v) : data(std::move(v)) {}

    bool is_null() const { return std::holds_alternative<std::nullptr_t>(data); }
    bool is_bool() const { return std::holds_alternative<bool>(data); }
    bool is_int() const { return std::holds_alternative<std::int64_t>(data); }
    bool is_double() const { return std::holds_alternative<double>(data); }
    bool is_number() const { return is_int() || is_double(); }
    bool is_string() const { return std::holds_alternative<std::string>(data); }
    bool is_array() const { return std::holds_alternative<Array>(data); }
    bool is_object() const { return std::holds_alternative<Object>(data); }

    bool as_bool() const {
        if (!is_bool()) throw std::runtime_error("JSON value is not a boolean");
        return std::get<bool>(data);
    }

    std::int64_t as_int() const {
        if (!is_int()) throw std::runtime_error("JSON value is not an integer");
        return std::get<std::int64_t>(data);
    }

    double as_double() const {
        if (is_double()) return std::get<double>(data);
        if (is_int()) return static_cast<double>(std::get<std::int64_t>(data));
        throw std::runtime_error("JSON value is not a number");
    }

    const std::string& as_string() const {
        if (!is_string()) throw std::runtime_error("JSON value is not a string");
        return std::get<std::string>(data);
    }

    const Array& as_array() const {
        if (!is_array()) throw std::runtime_error("JSON value is not an array");
        return std::get<Array>(data);
    }

    const Object& as_object() const {
        if (!is_object()) throw std::runtime_error("JSON value is not an object");
        return std::get<Object>(data);
    }

    bool contains(const std::string& key) const {
        if (!is_object()) return false;
        const auto& obj = std::get<Object>(data);
        return obj.find(key) != obj.end();
    }

    const Value& at(const std::string& key) const {
        const auto& obj = as_object();
        const auto it = obj.find(key);
        if (it == obj.end()) {
            throw std::runtime_error("Missing required JSON key: " + key);
        }
        return it->second;
    }
};

namespace detail {

class Parser {
public:
    explicit Parser(const std::string& input) : src_(input), pos_(0) {}

    Value parse_root() {
        skip_ws();
        Value root = parse_value();
        skip_ws();
        if (pos_ != src_.size()) {
            throw error("Unexpected trailing characters after JSON document");
        }
        return root;
    }

private:
    const std::string& src_;
    std::size_t pos_;

    std::runtime_error error(const std::string& msg) const {
        return std::runtime_error("JSON parse error at offset " + std::to_string(pos_) + ": " + msg);
    }

    bool eof() const { return pos_ >= src_.size(); }
    char peek() const { return eof() ? '\0' : src_[pos_]; }
    char get() {
        if (eof()) throw error("Unexpected end of input");
        return src_[pos_++];
    }

    void skip_ws() {
        while (!eof()) {
            const char c = src_[pos_];
            if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
                ++pos_;
            } else {
                break;
            }
        }
    }

    Value parse_value() {
        skip_ws();
        if (eof()) throw error("Unexpected end of input while expecting value");
        const char c = peek();
        if (c == '{') return parse_object();
        if (c == '[') return parse_array();
        if (c == '"') return Value(parse_string());
        if (c == 't') return parse_literal("true", Value(true));
        if (c == 'f') return parse_literal("false", Value(false));
        if (c == 'n') return parse_literal("null", Value(nullptr));
        if (c == '-' || (c >= '0' && c <= '9')) return parse_number();
        throw error(std::string("Unexpected character '") + c + "'");
    }

    Value parse_literal(const char* lit, Value result) {
        for (std::size_t i = 0; lit[i] != '\0'; ++i) {
            if (eof() || src_[pos_] != lit[i]) {
                throw error(std::string("Expected literal '") + lit + "'");
            }
            ++pos_;
        }
        return result;
    }

    static void append_utf8(std::string& out, std::uint32_t cp) {
        if (cp <= 0x7Fu) {
            out.push_back(static_cast<char>(cp));
        } else if (cp <= 0x7FFu) {
            out.push_back(static_cast<char>(0xC0u | (cp >> 6)));
            out.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
        } else if (cp <= 0xFFFFu) {
            out.push_back(static_cast<char>(0xE0u | (cp >> 12)));
            out.push_back(static_cast<char>(0x80u | ((cp >> 6) & 0x3Fu)));
            out.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
        } else {
            out.push_back(static_cast<char>(0xF0u | (cp >> 18)));
            out.push_back(static_cast<char>(0x80u | ((cp >> 12) & 0x3Fu)));
            out.push_back(static_cast<char>(0x80u | ((cp >> 6) & 0x3Fu)));
            out.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
        }
    }

    std::uint32_t parse_hex4() {
        std::uint32_t val = 0;
        for (int i = 0; i < 4; ++i) {
            const char c = get();
            val <<= 4;
            if (c >= '0' && c <= '9') val |= static_cast<std::uint32_t>(c - '0');
            else if (c >= 'a' && c <= 'f') val |= static_cast<std::uint32_t>(10 + (c - 'a'));
            else if (c >= 'A' && c <= 'F') val |= static_cast<std::uint32_t>(10 + (c - 'A'));
            else throw error("Invalid hex digit in \\u escape");
        }
        return val;
    }

    std::string parse_string() {
        if (get() != '"') throw error("Expected '\"'");
        std::string out;
        while (!eof()) {
            const char c = get();
            if (c == '"') return out;
            if (static_cast<unsigned char>(c) < 0x20u) {
                throw error("Unescaped control character in string");
            }
            if (c == '\\') {
                const char esc = get();
                switch (esc) {
                    case '"':  out.push_back('"'); break;
                    case '\\': out.push_back('\\'); break;
                    case '/':  out.push_back('/'); break;
                    case 'b':  out.push_back('\b'); break;
                    case 'f':  out.push_back('\f'); break;
                    case 'n':  out.push_back('\n'); break;
                    case 'r':  out.push_back('\r'); break;
                    case 't':  out.push_back('\t'); break;
                    case 'u': {
                        std::uint32_t cp = parse_hex4();
                        if (cp >= 0xD800u && cp <= 0xDBFFu) {
                            if (pos_ + 1 < src_.size() && src_[pos_] == '\\' && src_[pos_ + 1] == 'u') {
                                pos_ += 2;
                                const std::uint32_t low = parse_hex4();
                                if (low < 0xDC00u || low > 0xDFFFu) {
                                    throw error("Invalid UTF-16 low surrogate");
                                }
                                cp = 0x10000u + (((cp - 0xD800u) << 10) | (low - 0xDC00u));
                            } else {
                                throw error("Missing UTF-16 low surrogate");
                            }
                        }
                        append_utf8(out, cp);
                        break;
                    }
                    default:
                        throw error(std::string("Invalid escape sequence '\\") + esc + "'");
                }
            } else {
                out.push_back(c);
            }
        }
        throw error("Unterminated string");
    }

    Value parse_number() {
        const std::size_t start = pos_;
        bool is_float = false;

        if (peek() == '-') ++pos_;
        if (eof()) throw error("Incomplete number");

        if (peek() == '0') {
            ++pos_;
        } else if (peek() >= '1' && peek() <= '9') {
            while (!eof() && peek() >= '0' && peek() <= '9') ++pos_;
        } else {
            throw error("Invalid number");
        }

        if (!eof() && peek() == '.') {
            is_float = true;
            ++pos_;
            if (eof() || peek() < '0' || peek() > '9') {
                throw error("Expected digit after decimal point");
            }
            while (!eof() && peek() >= '0' && peek() <= '9') ++pos_;
        }

        if (!eof() && (peek() == 'e' || peek() == 'E')) {
            is_float = true;
            ++pos_;
            if (!eof() && (peek() == '+' || peek() == '-')) ++pos_;
            if (eof() || peek() < '0' || peek() > '9') {
                throw error("Expected digit in exponent");
            }
            while (!eof() && peek() >= '0' && peek() <= '9') ++pos_;
        }

        const std::string token = src_.substr(start, pos_ - start);
        if (!is_float) {
            try {
                std::size_t consumed = 0;
                const std::int64_t iv = std::stoll(token, &consumed, 10);
                if (consumed == token.size()) {
                    return Value(iv);
                }
            } catch (const std::out_of_range&) {
                // Fall through to double for values outside int64 range.
            }
        }

        char* endptr = nullptr;
        const double dv = std::strtod(token.c_str(), &endptr);
        if (endptr != token.c_str() + token.size()) {
            throw error("Failed to parse number: " + token);
        }
        return Value(dv);
    }

    Value parse_array() {
        if (get() != '[') throw error("Expected '['");
        Array arr;
        skip_ws();
        if (!eof() && peek() == ']') {
            ++pos_;
            return Value(std::move(arr));
        }
        while (true) {
            arr.push_back(parse_value());
            skip_ws();
            if (eof()) throw error("Unterminated array");
            const char c = get();
            if (c == ']') break;
            if (c != ',') throw error("Expected ',' or ']' in array");
        }
        return Value(std::move(arr));
    }

    Value parse_object() {
        if (get() != '{') throw error("Expected '{'");
        Object obj;
        skip_ws();
        if (!eof() && peek() == '}') {
            ++pos_;
            return Value(std::move(obj));
        }
        while (true) {
            skip_ws();
            if (eof() || peek() != '"') throw error("Expected string key in object");
            std::string key = parse_string();
            skip_ws();
            if (eof() || get() != ':') throw error("Expected ':' after object key");
            Value val = parse_value();
            obj[std::move(key)] = std::move(val);
            skip_ws();
            if (eof()) throw error("Unterminated object");
            const char c = get();
            if (c == '}') break;
            if (c != ',') throw error("Expected ',' or '}' in object");
        }
        return Value(std::move(obj));
    }
};

} // namespace detail

inline Value parse(const std::string& input) {
    detail::Parser parser(input);
    return parser.parse_root();
}

} // namespace mini_json
} // namespace bsptc
