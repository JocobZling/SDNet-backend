package cn.sd.utils;

import io.jsonwebtoken.Jwts;
import io.jsonwebtoken.SignatureAlgorithm;
import org.json.JSONObject;

import java.io.UnsupportedEncodingException;
import java.util.Map;

public class JwtUtil {
    private static final String SECRET = "XSGSecret";

    public static Map<String, Object> parseTokenToMap(String token) {
        return (Map<String, Object>) Jwts.parser()
                .setSigningKey(SECRET.getBytes())
                .parse(token)
                .getBody();
    }

    public static String build(Object obj) throws UnsupportedEncodingException {
        JSONObject jsonObject = new JSONObject(obj);
        return Jwts.builder()
                .setSubject(jsonObject.toString())
                .signWith(SignatureAlgorithm.HS512, SECRET.getBytes("UTF-8"))
                .compact();
    }
}
