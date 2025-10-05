
using UnityEngine;
using WebSocketSharp;
using System;
using System.Collections.Generic;

[Serializable] public class Marker { public int id; public float[] px; public float[] uv; }
[Serializable] public class Payload { public List<Marker> markers; public double ts; }

public class WSClient : MonoBehaviour {
    public string wsUrl = "ws://127.0.0.1:8765";
    WebSocket ws;
    void Start() {
        ws = new WebSocket(wsUrl);
        ws.OnMessage += (s, e) => {
            try {
                var p = JsonUtility.FromJson<Payload>(e.Data);
                if (p != null && p.markers != null) {
                    foreach (var m in p.markers) {
                        string uv = (m.uv != null && m.uv.Length==2) ? $"({m.uv[0]:F3},{m.uv[1]:F3})" : "None";
                        Debug.Log($"Marker {m.id} px=({m.px[0]:F1},{m.px[1]:F1}) uv={uv}");
                        // TODO: Map uv to regions and trigger effects.
                    }
                }
            } catch (Exception ex) { Debug.LogWarning("Parse failed: "+ex.Message); }
        };
        ws.ConnectAsync();
    }
    void OnDestroy(){ try{ if(ws!=null && ws.IsAlive) ws.CloseAsync(); }catch{} }
}
