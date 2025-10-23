
using UnityEngine;
using WebSocketSharp;
using System;
using System.Collections.Generic;
using System.Collections.Concurrent;

[Serializable] public class Marker { public int id; public float[] px; public float[] uv; }
[Serializable] public class MapInfo { public float[] size; }
[Serializable] public class Payload { public List<Marker> markers; public double ts; public MapInfo map; public bool hasH; }

public class WSClient : MonoBehaviour {
    public string wsUrl = "ws://127.0.0.1:8765";
    WebSocket ws;
    readonly ConcurrentQueue<string> messageQueue = new ConcurrentQueue<string>();
    double lastTs = -1.0;

    void Start() {
        ws = new WebSocket(wsUrl);
        ws.OnOpen += (s, e) => Debug.Log("WebSocket connected");
        ws.OnError += (s, e) => Debug.LogWarning("WebSocket error: " + e.Message);
        ws.OnMessage += (s, e) => messageQueue.Enqueue(e.Data);
        ws.ConnectAsync();
    }

    void Update() {
        while (messageQueue.TryDequeue(out var json)) {
            try {
                var payload = JsonUtility.FromJson<Payload>(json);
                if (payload == null || payload.markers == null) {
                    continue;
                }

                if (lastTs > 0 && payload.ts > lastTs) {
                    Debug.Log($"Δt={payload.ts - lastTs:F3}s markers={payload.markers.Count}");
                }
                lastTs = payload.ts;

                foreach (var marker in payload.markers) {
                    string uv = (marker.uv != null && marker.uv.Length == 2)
                        ? $"({marker.uv[0]:F3},{marker.uv[1]:F3})"
                        : "None";
                    Debug.Log($"Marker {marker.id} px=({marker.px[0]:F1},{marker.px[1]:F1}) uv={uv}");
                }
            } catch (Exception ex) {
                Debug.LogWarning("Parse failed: " + ex.Message);
            }
        }
    }

    void OnDestroy(){ try{ if(ws!=null && ws.IsAlive) ws.CloseAsync(); }catch{} }
}
