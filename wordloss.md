# **A Strategic Analysis and Resolution for Word Loss in Real-Time Transcription Connection Switching**

## **A Critical Analysis of the 31-Second Connection Rotation Strategy**

The current architecture, which rotates WebSocket connections to the
Speechmatics service every 31 seconds, is founded on a premise that
requires careful examination. The stated goal is to ensure "connection
stability," but this proactive rotation introduces significant
complexity and is the direct cause of the word loss it seeks to prevent.
A thorough analysis of the Speechmatics API's documented behavior and
common networking principles reveals that this strategy may be a
solution to a misidentified problem.

### **Validating the Underlying Premise: Is Frequent Rotation Necessary?**

The decision to implement a high-frequency connection-swapping mechanism
appears to be based on a misunderstanding of the Speechmatics real-time
service's design for session longevity. The service is explicitly
engineered for long-duration, continuous use. Documentation confirms
that real-time SaaS sessions can remain active for up to **48 hours**
before an automatic termination occurs.<sup>1</sup> Furthermore, session
termination due to inactivity is only triggered under specific, generous
conditions: no audio data being sent for a full hour, or a complete
absence of both audio and keepalive messages (ping/pongs) for three
minutes.<sup>1</sup>

These operational parameters stand in stark contrast to the 31-second
lifecycle imposed by the current script. The API is not designed to be
unstable on such a short timescale; rather, it is built for robust,
persistent connections. This discrepancy suggests that the root cause of
any perceived instability is not an inherent limitation of the
Speechmatics service. Instead, it is highly probable that the system is
experiencing intermittent disconnects from more common sources, such as
network intermediaries. Firewalls, NAT gateways, and load balancers are
frequently configured to terminate TCP connections that they perceive as
idle, often with timeouts as short as 30 to 120 seconds.<sup>2</sup>
Without a mechanism to signal that the connection is still in use, these
intermediaries can prematurely close the WebSocket session.

Therefore, the architectural choice to rotate connections every 31
seconds is not addressing a fault in the API but is likely a workaround
for a network-level issue that has not been correctly diagnosed. The
system treats the symptom—occasional, unexpected disconnections—rather
than the underlying cause, which is likely a lack of a proper keepalive
mechanism. This reframes the entire problem: the objective should not be
to perfect a complex and costly rotation scheme, but to build a single,
resilient connection that can withstand and gracefully recover from
infrequent, real-world network failures.

### **The True Cost of the Hot-Swap Architecture**

The user's own performance analysis correctly identifies several
significant costs associated with the active/backup connection model,
including increased task concurrency, higher memory consumption, and
elevated CPU usage. These are direct and unavoidable consequences of
maintaining two parallel WebSocket connections and the logic required to
manage them. However, the full cost extends beyond these immediate
resource metrics.

- **Increased API Overhead:** Each 31-second cycle involves a full
  WebSocket handshake and the transmission of StartRecognition and
  EndOfStream messages.<sup>4</sup> This constant setup and teardown
  generates unnecessary traffic and places a higher load on both client
  and server resources compared to a single, long-lived connection.

- **Concurrency and Billing Risks:** While Speechmatics bills per second
  of audio processed <sup>6</sup>, the frequent creation of new
  connections increases the risk of inadvertently exceeding account
  concurrency limits, especially if the cleanup process for old
  connections fails or is delayed. The Free Tier is limited to 2
  concurrent connections, and the Paid Tier to 20, making this a
  tangible risk.<sup>1</sup> Failed handoffs could lead to "zombie"
  connections that consume valuable concurrency slots.

- **Architectural Brittleness:** The complexity of orchestrating two
  parallel streams, managing their state, and synchronizing a "critical
  moment" swap introduces multiple potential points of failure. Race
  conditions during the swap, errors in the backup connection's
  initialization, or failures in the cleanup task can lead to states
  that are difficult to debug and recover from, ultimately reducing the
  system's overall reliability. The current architecture, intended to
  increase stability, paradoxically introduces new vectors for
  instability.

## **The Simplest Path to Stability: The Robust Single Connection Architecture**

The most direct and effective solution is to abandon the frequent
rotation strategy in favor of a single, resilient connection. This
approach aligns with standard industry best practices for long-lived
WebSockets and directly addresses the likely root causes of instability
while eliminating the complexity and word-loss issues inherent in the
current design. The guiding principle is to shift from proactive,
stateful redundancy to reactive, stateless resilience.

### **Implementing Protocol-Level Keepalives for Proactive Health Monitoring**

As identified, a primary suspect for connection drops is idle timeouts
imposed by network infrastructure.<sup>2</sup> The WebSocket protocol
provides a built-in solution for this: Ping and Pong control frames. By
periodically sending a Ping frame, the client can achieve two critical
objectives:

1.  **Keepalive:** The small amount of traffic generated by the
    Ping/Pong exchange prevents network intermediaries from classifying
    the connection as idle and prematurely closing it.

2.  **Heartbeat:** If a Pong frame is not received within a specified
    timeout, the client can definitively conclude that the connection is
    broken and initiate recovery procedures. This is far more reliable
    than waiting for a TCP timeout, which can take several
    minutes.<sup>2</sup>

Speechmatics explicitly recommends this practice, suggesting a ping
interval of 20 to 60 seconds and a ping timeout of at least 60
seconds.<sup>1</sup> Implementing this is a simple configuration change
in the Python

websockets library and is likely to resolve the majority of perceived
instability issues with minimal effort.

### **Designing a Resilient ConnectionManager with Automatic Reconnection**

Even with keepalives, connections can fail for legitimate reasons, such
as the 48-hour session limit, a server-side maintenance event, or a
genuine network outage.<sup>1</sup> A robust application must handle
these events gracefully rather than crashing. The standard pattern for
this is an automatic reconnection loop with exponential backoff.

Upon detecting a closed connection (e.g., by catching a
websockets.exceptions.ConnectionClosed exception), the client should not
attempt to reconnect immediately. Doing so can lead to "thrashing,"
where the client repeatedly hammers a server that is temporarily
unavailable, wasting resources on both ends.<sup>8</sup> Instead, the
client should wait for a short period before retrying. If the second
attempt also fails, the waiting period should be increased (e.g.,
doubled). This exponential increase in delay continues up to a defined
maximum, giving the server or network time to recover.<sup>9</sup> This
is a standard, battle-tested strategy for building resilient network
clients.<sup>11</sup>

### **Code Implementation: A Resilient SpeechmaticsConnection**

The principles of keepalives and automatic reconnection can be
integrated into the existing class structure. The ConnectionManager
would be simplified to manage only a single SpeechmaticsConnection and
would be responsible for the reconnection loop.

The main execution logic within the ConnectionManager would be
transformed from a timed swap loop to a resilience loop:

> Python

import asyncio  
import websockets  
import random  
  
\# Simplified ConnectionManager logic  
async def manage_connection_loop(self):  
"""Manages a single, resilient connection with exponential backoff."""  
reconnect_delay = 1.0 \# Initial delay in seconds  
max_reconnect_delay = 60.0  
  
while True:  
try:  
\# The connect() method should now handle the full lifecycle  
await self.active_connection.connect_and_process()  
  
\# If connect_and_process returns cleanly (e.g., graceful shutdown)  
\# Reset delay and break or continue as needed  
reconnect_delay = 1.0  
print("Connection closed gracefully. Exiting.")  
break  
  
except websockets.exceptions.ConnectionClosed as e:  
print(f"Connection closed unexpectedly: {e}. Reconnecting...")  
except Exception as e:  
print(f"An unexpected error occurred: {e}. Reconnecting...")  
  
\# Exponential backoff with jitter  
jitter = random.uniform(0.5, 1.5)  
sleep_duration = min(reconnect_delay \* jitter, max_reconnect_delay)  
print(f"Will attempt to reconnect in {sleep_duration:.2f} seconds.")  
await asyncio.sleep(sleep_duration)  
reconnect_delay \*= 2

The SpeechmaticsConnection.connect_and_process method would encapsulate
the connection and message handling, now including the vital keepalive
parameters.

> Python

\# Simplified SpeechmaticsConnection method  
async def connect_and_process(self):  
"""Connects with keepalives and processes messages."""  
\# Speechmatics recommends ping_interval=20-60s, ping_timeout\>=60s  
async with websockets.connect(  
self.url,  
ping_interval=20,  
ping_timeout=60  
) as websocket:  
self.websocket = websocket  
  
\# Send StartRecognition message  
await self.send_start_recognition()  
  
\# Concurrently send audio and receive messages  
\# This part of the logic would contain the audio sending  
\# and message receiving tasks.  
\#... (user's existing audio sender and message receiver logic)...

This simplified, robust architecture directly solves the user's request
for a "simple and working solution." It eliminates word loss by removing
the connection swap, improves performance by reducing overhead, and
increases stability by implementing industry-standard resilience
patterns.

## **The Definitive Guide to Seamless Handoff: A Dual-Stream Overlap & Merge Strategy**

While the single-connection architecture is strongly recommended, there
may be external constraints that make the 31-second rotation a
non-negotiable requirement. In this specific scenario, the goal must be
to transform the current "hot-swap" mechanism, which is inherently
lossy, into a "warm-handoff" protocol that guarantees transcript
continuity. This is achieved through a brief period of dual audio
streaming and a precise, timestamp-based transcript merge.

### **Architectural Blueprint: From "Hot-Swap" to "Warm-Handoff"**

The core flaw in the current system is the instantaneous redirection of
audio. The new connection has no context and requires a brief "warm-up"
period to begin processing accurately, during which words are lost. The
solution is to create an **overlap period** of 2-3 seconds where both
the old (soon-to-be-retired) connection and the new (soon-to-be-active)
connection receive the exact same audio stream.<sup>12</sup> This
buffered audio gives the new connection the time it needs to initialize
and start transcribing before it becomes the sole active connection.
This process combines the user's potential solutions of "Audio
Buffering" and "Dual Processing" into a single, cohesive strategy.

### **Phase 1: The Overlap Period and Verified Handoff**

A robust handoff cannot be a "blind swap." The system must verify that
the new connection is not only established but is actively processing
audio before committing to the switch. This creates a transactional,
reliable handoff.

The process begins several seconds before the scheduled swap time (T=0):

1.  **Initiate Backup (T-3s):** The ConnectionManager establishes the
    WebSocket for the new backup connection and sends the
    StartRecognition message.

2.  **Start Overlap and Verify Liveness (T-2s):** The manager begins
    duplicating the live audio stream, sending it to *both* the active
    and backup connections. It then critically waits to receive a
    AddPartialTranscript message from the backup connection. Partial
    transcripts are the earliest indicator of a healthy, processing
    stream, typically arriving in under 500ms.<sup>14</sup>

3.  **Commit or Abort:** If a partial transcript is received from the
    backup connection within a set timeout (e.g., 2 seconds), the
    handoff is verified and can proceed. If no partial is received, the
    backup is assumed to be faulty, the handoff is aborted, and the
    original connection remains active. This verification step prevents
    swapping to a dead or malfunctioning connection, a critical failure
    mode in the current architecture.

### **Phase 2: The Merge Algorithm (Stitching Transcripts with Timestamp Precision)**

Once the handoff is committed at T=0, the two transcript streams must be
merged into one seamless whole. This is the lynchpin of the lossless
strategy and is made possible by a key feature of the Speechmatics API:
the AddTranscript message contains word-level start_time and end_time
timestamps.<sup>7</sup> These timestamps provide the precise coordinates
needed for a perfect merge.

The algorithm proceeds as follows:

1.  **Finalize Old Stream:** After the swap, the old connection is sent
    an EndOfStream message. The system allows it a grace period (e.g.,
    max_delay + 1 second) to process its remaining audio buffer and send
    its final AddTranscript messages. All words from the old connection
    are collected into a list, old_words, preserving their timestamps.

2.  **Identify Handoff Point:** The exact end_time of the very last word
    in the old_words list is identified. This becomes the
    handoff_timestamp.

3.  **Process and Filter New Stream:** As the new, now-active connection
    produces AddTranscript messages, its words are collected. Each word
    from this new stream is checked against the handoff_timestamp.

4.  **Stitch Transcripts:** Any word from the new stream whose
    start_time is less than or equal to the handoff_timestamp is
    discarded. This deterministically removes any duplicate words that
    were transcribed by both connections during the overlap period. The
    remaining words from the new stream are then appended to the
    old_words list.

This process is a practical application of sequence alignment, where
time is the axis of alignment and the established stream (the old
connection) is given precedence for any overlapping
segments.<sup>16</sup> The result is a single, continuous transcript
with no gaps and no duplications.

### **Table: The Warm Handoff & Transcript Merging Process**

The following table illustrates the warm handoff process for the spoken
phrase "This is the final sentence that gets processed." The swap occurs
after the word "gets".

| Time Step | Audio Spoken | Connection 1 (Old Active) State | Connection 2 (New Active) State | Merge Logic Action | Final User-Facing Transcript |
|----|----|----|----|----|----|
| T-2s | "sentence that" | Processing audio. Sends AddTranscript for "sentence", "that". | Receiving duplicate audio stream. Initializing. | Buffer words from Connection 1. | "This is the final sentence that" |
| T-1s | "gets" | Processing audio. Sends AddTranscript for "gets". | Receiving duplicate audio. Sends first AddPartialTranscript. | **Handoff Verified.** Buffer word from Connection 1. | "This is the final sentence that gets" |
| **T=0** | **(SWAP)** | EndOfStream sent. Processing final buffer. | Becomes primary. Receives "processed" audio directly. | **Swap complete.** Wait for final words from Connection 1. | "This is the final sentence that gets" |
| T+1s | "processed" | Sends final AddTranscript for "gets". Last word end_time is t_final. | Processing audio. Sends AddTranscript for "gets", "processed". | Store handoff_timestamp = t_final. | "This is the final sentence that gets" |
| T+2s | (silence) | Connection closed. | Processing audio. Sends AddTranscript for "processed". | Discard "gets" from Conn 2 (start_time \< t_final). Keep "processed" (start_time \> t_final). | **"This is the final sentence that gets processed"** |

## **Implementation Blueprint: Refactoring for Lossless Transcription**

To implement the warm handoff strategy, modifications are required in
both the SpeechmaticsConnection and ConnectionManager classes to handle
the new lifecycle of overlapping audio and merging transcripts.

### **Modifying the SpeechmaticsConnection Class**

The SpeechmaticsConnection class must be enhanced to support the merge
algorithm. It can no longer simply concatenate transcript strings into a
global variable.

- **Internal State:** The class needs an internal list to store the full
  AddTranscript message objects received from the API. Each object
  should contain the word content, start_time, and end_time.

- **New Methods:**

  - get_final_words(): A method that can be called by the
    ConnectionManager after the connection is terminated. It returns the
    complete, ordered list of word objects with their associated
    metadata.

  - clear_transcript_buffer(): A method to reset the internal list of
    words, to be called before a new connection is used.

### **Refactoring the ConnectionManager for the Overlap & Merge Lifecycle**

The ConnectionManager requires the most significant changes, as it
orchestrates the entire handoff process.

- **Replacing \_swap_connections:** The simple, instantaneous swap
  method must be replaced with a more sophisticated
  \_execute_warm_handoff asynchronous method that manages the entire
  lifecycle from verification to merging.

- **State Management:** The manager will need to control the audio
  duplication, manage the overlap period, and execute the merge logic.

- **Pseudocode for \_execute_warm_handoff:**  
  Python  
  \# High-level pseudocode within ConnectionManager  
  async def \_execute_warm_handoff(self):  
  print("Starting warm handoff...")  
    
  \# 1. Create backup and start its message receiver task  
  new_conn = self.create_backup_connection()  
  verification_event = asyncio.Event()  
  asyncio.create_task(new_conn.message_receiver(verification_event))  
    
  \# 2. Start overlap period: send audio to both  
  self.backup_connection = new_conn  
  self.is_duplicating_audio = True  
    
  \# 3. Verified Handoff: wait for first partial from backup  
  try:  
  await asyncio.wait_for(verification_event.wait(), timeout=3.0)  
  print("Handoff verified: backup connection is live.")  
  except asyncio.TimeoutError:  
  print("Handoff verification failed. Aborting swap.")  
  await self.backup_connection.close()  
  self.backup_connection = None  
  self.is_duplicating_audio = False  
  return  
    
  \# 4. Perform the swap  
  old_conn = self.active_connection  
  self.active_connection = self.backup_connection  
  self.backup_connection = None \# Old backup is now the new active  
  self.is_duplicating_audio = False \# Stop duplication  
    
  \# 5. Graceful shutdown of old connection and merge transcripts  
  await self.\_cleanup_and_merge(old_conn, self.active_connection)  
  print("Handoff complete.")  
    
  async def \_cleanup_and_merge(self, old_conn, new_conn):  
  \# Signal old connection to finish  
  await old_conn.send_end_of_stream()  
  \# Allow time for final messages  
  await asyncio.sleep(self.config.max_delay + 1.0)  
  await old_conn.close()  
    
  \# Get word lists from both connections  
  old_words = old_conn.get_final_words()  
    
  if not old_words:  
  return \# No merge needed if old connection had no words  
    
  \# Execute the timestamp-based merge algorithm  
  handoff_timestamp = old_words\[-1\]\['end_time'\]  
  new_conn.set_merge_start_time(handoff_timestamp)  
  \# The new_conn's message handler will now filter words based on this
  timestamp

### **Optimizing transcription_config for Faster Handoffs**

The transcription_config sent in the StartRecognition message can be
tuned to minimize the warm-up and overlap period.

- **max_delay:** For the new backup connection, setting a lower
  max_delay, such as 1.0 or 1.5 seconds, will cause it to generate
  AddTranscript messages more quickly. This allows for faster
  confirmation of liveness and reduces the amount of audio that needs to
  be buffered. While this may slightly reduce accuracy for the first few
  words, it is an acceptable trade-off for a smoother
  transition.<sup>15</sup>

- **enable_partials:** This must be set to true for the backup
  connection. Receiving a partial transcript is the trigger for the
  "Verified Handoff" and is essential for the reliability of the
  swap.<sup>14</sup>

- **Dynamic Configuration:** The API supports a SetRecognitionConfig
  message, which allows for modifying parameters
  mid-session.<sup>4</sup> After a successful handoff, the system could
  send this message to the new active connection to increase  
  max_delay back to a higher-accuracy value (e.g., 2.0 or 4.0 seconds)
  for normal operation.

## **Strategic Recommendations and Final Verdict**

The analysis reveals that the word-loss problem is not an unavoidable
flaw in the transcription service but a direct consequence of a
client-side architecture built on a flawed premise. The path to a
stable, lossless, and efficient system involves aligning the
application's design with the documented behavior of the Speechmatics
API and with standard network engineering practices.

### **Recommendation 1: Adopt the Robust Single Connection (The 99% Solution)**

**This is the primary and most strongly recommended course of action.**
It is the simplest, most performant, and most reliable solution. It
eradicates the root cause of the word-loss problem—the connection swap
itself—by moving to a single, long-lived connection model.

- **Implementation:**

  1.  Remove the entire active/backup connection management and the
      31-second refresh loop.

  2.  In the websockets.connect() call, add ping_interval=20 and
      ping_timeout=60 to enable the WebSocket keepalive mechanism.

  3.  Wrap the connection logic in a while True loop that catches
      websockets.exceptions.ConnectionClosed and implements an automatic
      reconnect with exponential backoff.

- **Outcome:** A vastly simplified, more robust, and more efficient
  application that provides continuous, lossless transcription.

### **Recommendation 2: Implement the Dual-Stream Handoff (For Unavoidable Rotations)**

This recommendation should only be considered if the 31-second rotation
is an unchangeable external requirement. This approach correctly
implements the user's original intent, guaranteeing lossless
transcription at the cost of maintaining higher architectural complexity
and resource overhead.

- **Implementation:**

  1.  Implement an audio overlap period of 2-3 seconds before the swap.

  2.  Institute a "Verified Handoff" by waiting for a partial transcript
      from the backup connection before committing the swap.

  3.  Implement a precise, timestamp-based merge algorithm to stitch the
      final words from the old connection to the initial words of the
      new connection.

- **Outcome:** A complex but functional system that achieves lossless
  transcription during frequent connection rotations.

### **Table: Comparative Analysis of Proposed Solutions**

| Feature | Current Hot-Swap Architecture | **Robust Single Connection (Rec. 1)** | Dual-Stream Warm Handoff (Rec. 2) |
|----|----|----|----|
| **Architectural Complexity** | High | **Low** | Very High |
| **CPU/Memory Overhead** | High | **Low** | High |
| **Code Maintainability** | Low | **High** | Low |
| **Lossless Guarantee** | No (inherently lossy) | **Yes** | Yes |
| **Primary Failure Addressed** | Misunderstood API Instability | **Real-world Network Failures** | Required, Frequent Connection Cycling |

### **Concluding Thoughts: A Path to a Simpler, More Effective System**

The investigation concludes that the most effective path forward is to
challenge the initial assumption that frequent connection rotation is
necessary. By embracing the stability inherent in the Speechmatics API
and building a single, resilient connection, the system can be
simplified dramatically while simultaneously solving the word-loss
problem and improving overall performance. The first and most critical
step is to re-evaluate the 31-second rotation requirement. Removing it
opens the door to a far more robust, efficient, and maintainable
solution.

#### Works cited

1.  Supported Files and Limits - Speechmatics, accessed on July 24,
    2025,
    [<u>https://docs.speechmatics.com/introduction/errors-rate-limits</u>](https://docs.speechmatics.com/introduction/errors-rate-limits)

2.  Keepalive and latency - websockets 15.0.1 documentation, accessed on
    July 24, 2025,
    [<u>https://websockets.readthedocs.io/en/stable/topics/keepalive.html</u>](https://websockets.readthedocs.io/en/stable/topics/keepalive.html)

3.  WebSocket architecture best practices: Designing scalable realtime
    systems - Ably, accessed on July 24, 2025,
    [<u>https://ably.com/topic/websocket-architecture-best-practices</u>](https://ably.com/topic/websocket-architecture-best-practices)

4.  Websocket API Reference - Getting started - Speechmatics, accessed
    on July 24, 2025,
    [<u>https://legacy.docs.speechmatics.com/en/real-time-appliance/api-v2/speech-api-guide/v3.4.0</u>](https://legacy.docs.speechmatics.com/en/real-time-appliance/api-v2/speech-api-guide/v3.4.0)

5.  Websocket API Reference - Getting started, accessed on July 24,
    2025,
    [<u>https://legacy.docs.speechmatics.com/en/real-time-appliance/api-v2/speech-api-guide/v4.0.0</u>](https://legacy.docs.speechmatics.com/en/real-time-appliance/api-v2/speech-api-guide/v4.0.0)

6.  Real-Time - Speechmatics, accessed on July 24, 2025,
    [<u>https://www.speechmatics.com/product/real-time</u>](https://www.speechmatics.com/product/real-time)

7.  Real-Time API Reference - Speechmatics, accessed on July 24, 2025,
    [<u>https://docs.speechmatics.com/rt-api-ref</u>](https://docs.speechmatics.com/rt-api-ref)

8.  WebSocket Reconnect: Strategies for Reliable Communication - Apidog,
    accessed on July 24, 2025,
    [<u>https://apidog.com/blog/websocket-reconnect/</u>](https://apidog.com/blog/websocket-reconnect/)

9.  autobahn-python/examples/twisted/websocket/reconnecting/README.md at
    master, accessed on July 24, 2025,
    [<u>https://github.com/crossbario/autobahn-python/blob/master/examples/twisted/websocket/reconnecting/README.md</u>](https://github.com/crossbario/autobahn-python/blob/master/examples/twisted/websocket/reconnecting/README.md)

10. Re: Client reconnecting from Websockets in Python, accessed on July
    24, 2025,
    [<u>https://groups.google.com/g/autobahnws/c/pOv0t26qZ64</u>](https://groups.google.com/g/autobahnws/c/pOv0t26qZ64)

11. Client (legacy asyncio) - websockets 13.0.1 documentation, accessed
    on July 24, 2025,
    [<u>https://websockets.readthedocs.io/en/13.0.1/reference/legacy/client.html</u>](https://websockets.readthedocs.io/en/13.0.1/reference/legacy/client.html)

12. Real-Time Transcription Latency: What It Is and How to Optimize -
    AMC Technology, accessed on July 24, 2025,
    [<u>https://www.amctechnology.com/resources/blog/real-time-transcription-speed-latency</u>](https://www.amctechnology.com/resources/blog/real-time-transcription-speed-latency)

13. Really Real Time Speech To Text · openai whisper · Discussion
    \#608 - GitHub, accessed on July 24, 2025,
    [<u>https://github.com/openai/whisper/discussions/608</u>](https://github.com/openai/whisper/discussions/608)

14. Transcribe in Real-Time - Speechmatics, accessed on July 24, 2025,
    [<u>https://docs.speechmatics.com/introduction/rt-guide</u>](https://docs.speechmatics.com/introduction/rt-guide)

15. Real-Time Latency \| Speechmatics, accessed on July 24, 2025,
    [<u>https://docs.speechmatics.com/features/realtime-latency</u>](https://docs.speechmatics.com/features/realtime-latency)

16. Merge Overlapping Pairs - QIAGEN Bioinformatics Manuals, accessed on
    July 24, 2025,
    [<u>https://resources.qiagenbioinformatics.com/manuals/clcgenomicsworkbench/2301/index.php?manual=Merge_Overlapping_Pairs.html</u>](https://resources.qiagenbioinformatics.com/manuals/clcgenomicsworkbench/2301/index.php?manual=Merge_Overlapping_Pairs.html)

17. Sequence alignments — Biopython 1.86.dev0 documentation, accessed on
    July 24, 2025,
    [<u>https://biopython.org/docs/dev/Tutorial/chapter_align.html</u>](https://biopython.org/docs/dev/Tutorial/chapter_align.html)
