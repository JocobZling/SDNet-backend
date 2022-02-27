package cn.sd.utils;

import net.sf.json.JSONObject;
import org.apache.commons.codec.Charsets;
import org.apache.http.HttpResponse;
import org.apache.http.HttpStatus;
import org.apache.http.client.HttpClient;
import org.apache.http.client.methods.HttpGet;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.client.utils.URIBuilder;
import org.apache.http.entity.StringEntity;
import org.apache.http.impl.client.HttpClientBuilder;
import org.apache.http.util.EntityUtils;

import java.io.IOException;

public class HttpUtil {
    public static JSONObject get(String url) throws IOException {
        HttpClient httpClient = HttpClientBuilder.create().build();
        HttpGet httpGet = new HttpGet(url);
        HttpResponse response = httpClient.execute(httpGet);
        if (response.getStatusLine().getStatusCode() == HttpStatus.SC_OK) {
            String data = EntityUtils.toString(response.getEntity(), Charsets.UTF_8);
            JSONObject jsonObject = JSONObject.fromObject(data);
            return jsonObject;
        }
        httpGet.releaseConnection();
        return null;
    }

    public static JSONObject post(String url, String data) throws Exception {
        HttpClient httpClient = HttpClientBuilder.create().build();
        URIBuilder urlBuilder = new URIBuilder(url);
        HttpPost httpPost = new HttpPost(urlBuilder.build());
        httpPost.setHeader("Content-Type", "application/json;charset=utf-8");
        if (data != null) {
            StringEntity s = new StringEntity(data, "utf-8");
            s.setContentEncoding("utf-8");
            s.setContentType("application/json;charset=utf-8");//发送json数据需要设置contentType
            httpPost.setEntity(s);
        }

        HttpResponse response = httpClient.execute(httpPost);
        String result = EntityUtils.toString(response.getEntity(), Charsets.UTF_8);
        return JSONObject.fromObject(result);
    }
}
