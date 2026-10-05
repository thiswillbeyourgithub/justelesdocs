/* A bounded queue that runs one job at a time.
 *
 * `rank` in src/search.js is 40 to 60 ms of synchronous CPU on the full index, so
 * on one node thread it serialises itself whatever this file does. What the queue
 * adds is the bound: past `depth` waiting jobs a new one is refused at once with
 * "full" instead of joining a line nobody can see the end of, and a job whose
 * client went away before its turn is dropped with "gone" rather than ranked
 * for nobody. Both are the difference between a container that answers 503 under
 * a burst and one that stops answering at all (TODO.local.md, handoff A2).
 *
 * Written by Claude Code (Fable 5.1).
 */

/**
 * @param {{depth?: number}} [options] - How many jobs may WAIT; one more runs.
 * @returns {{run: (job: () => any, options?: {signal?: AbortSignal}) => Promise<any>,
 *   waiting: () => number, busy: () => boolean}}
 */
export function createQueue({ depth = 24 } = {}) {
  const waiting = [];
  let running = false;

  async function pump() {
    if (running) return;
    running = true;
    try {
      while (waiting.length) {
        // Let the event loop turn before every job. A job is synchronous CPU, so
        // without this the first request of a burst is ranked before the loop
        // has read the others off their sockets: they queue in the kernel, one
        // enters the line at a time, the line never fills, and every client of
        // the burst waits for the whole burst (measured, DESIGN.md: 50 at once,
        // 0 refused, 3.5 s each). The yield lets every request that has arrived
        // join the line first, so the bound holds and the rest get their 503
        // now instead of an answer in four seconds.
        await new Promise((turn) => setImmediate(turn));
        if (!waiting.length) break;
        const { job, signal, resolve, reject } = waiting.shift();
        if (signal?.aborted) {
          reject(new Error("gone"));
          continue;
        }
        try {
          resolve(await job());
        } catch (error) {
          reject(error);
        }
      }
    } finally {
      running = false;
    }
  }

  return {
    waiting: () => waiting.length,
    busy: () => running,

    /**
     * Queue `job` and settle with what it returns.
     *
     * @param {() => any} job - Run on its turn; may return a promise.
     * @param {{signal?: AbortSignal}} [options] - Aborted before its turn, the
     *   job never runs and the promise rejects with "gone".
     * @returns {Promise<any>} Rejects with "full" when the line is at `depth`.
     */
    run(job, { signal } = {}) {
      if (signal?.aborted) return Promise.reject(new Error("gone"));
      if (waiting.length >= depth) return Promise.reject(new Error("full"));
      return new Promise((resolve, reject) => {
        waiting.push({ job, signal, resolve, reject });
        void pump();
      });
    },
  };
}
