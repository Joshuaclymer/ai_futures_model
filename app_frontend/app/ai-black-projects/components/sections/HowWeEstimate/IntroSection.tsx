'use client';

import Image from 'next/image';

export function IntroSection() {
  return (
    <div className="how-we-estimate-intro">
      {/* Main heading */}
      <h2
        className="text-[28px] font-bold text-[#222] tracking-tight mb-4"
        style={{ letterSpacing: '-0.5px' }}
      >
        How we estimate covert computation
      </h2>

      {/* Introduction with diagram */}
      <div className="flex gap-8 items-start flex-wrap lg:flex-nowrap mb-6">
        {/* Text content */}
        <div
          className="flex-1 text-[15px] leading-[1.7] text-[#555]"
          style={{ maxWidth: '700px' }}
        >
          <p className="mb-4">Figure 1 summarizes our approach:</p>
          <ol className="list-decimal list-outside ml-5 space-y-3">
            <li>
              First, we create a model that predicts how quickly a covert project{' '}
              <strong>performs computation</strong> and its likelihood of being{' '}
              <strong>detected</strong> by US intelligence agencies.
            </li>
            <li>
              Then, we identify the strategic choices (such as how many people to involve)
              that maximize the total computation the project performs before detection.
              If a covert project is too large, it is detected quickly. If a covert project
              is too small, it makes little progress. So finding the optimal configuration
              requires striking a balance between size and stealth.
            </li>
            <li>
              Finally, we use the optimal{' '}
              <span className="text-[#5E6FB8] underline decoration-dotted cursor-pointer">
                project properties
              </span>{' '}
              identified in the previous step to come up with a worst-case estimate of how much{' '}
              <strong>covert computation</strong> the PRC might perform during an AI slowdown agreement.
            </li>
          </ol>
          <p className="mt-4">
            The next sections expand on the first step: how we mapped a{' '}
            <span className="text-[#5E6FB8] underline decoration-dotted cursor-pointer">
              project&apos;s properties
            </span>{' '}
            to its <strong>rate of computation</strong> and <strong>detection likelihood</strong>.
          </p>
        </div>

        {/* Diagram */}
        <div className="flex-1 flex flex-col items-center justify-start min-w-[300px]">
          <Image
            src="/model_diagram.svg"
            alt="Overview of the model"
            width={540}
            height={165}
            className="w-full max-w-[540px] h-auto"
            priority
          />
          <p className="text-center mt-2 italic text-[11px] text-[#666]">
            Figure 1. Overview of our forecasting methodology.
          </p>
        </div>
      </div>
    </div>
  );
}
